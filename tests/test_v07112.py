"""v0.71.12 — "Architecture + distill + adapter-train".

Closes #145 (cross-tokenizer / sequence-KD distill_mode), #146 (classifier
LoRA path), #148 (LLaMA Pro per-arch zero-init), #158 (LongLoRA S² shift on
Q/K projections), #84 (Mixture-of-Depths live patch), #221 (live VeRA/VB-LoRA
serving), #222 (live MoLE gating-kernel training + routing).

Heavy-model exercise lives in the Step-6 manual smoke (real tiny Llama / gpt2);
these tests use fake ``nn.Module`` shapes + schema round-trips so they run in
CI without a network download.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from soup_cli.config.loader import load_config_from_string


def _yaml_rejects(yaml: str, keyword: str) -> None:
    """Assert ``load_config_from_string`` rejects ``yaml`` naming ``keyword``."""
    with pytest.raises(Exception) as exc:  # noqa: PT011 — pydantic/ValueError union
        load_config_from_string(yaml)
    assert keyword in str(exc.value), f"expected {keyword!r} in: {exc.value}"


# ---------------------------------------------------------------------------
# #145 — distill_mode token | sequence
# ---------------------------------------------------------------------------


class TestDistillMode:
    def test_supported_modes_frozenset(self):
        from soup_cli.utils.distill import SUPPORTED_DISTILL_MODES

        assert SUPPORTED_DISTILL_MODES == frozenset({"token", "sequence"})
        assert isinstance(SUPPORTED_DISTILL_MODES, frozenset)

    @pytest.mark.parametrize("value,expected", [
        ("token", "token"), ("sequence", "sequence"),
        ("TOKEN", "token"), ("Sequence", "sequence"),
    ])
    def test_validate_happy(self, value, expected):
        from soup_cli.utils.distill import validate_distill_mode

        assert validate_distill_mode(value) == expected

    @pytest.mark.parametrize("bad", [True, 1, None, "", "tokens", "kl", "x" * 64, "to\x00ken"])
    def test_validate_rejects(self, bad):
        from soup_cli.utils.distill import validate_distill_mode

        with pytest.raises((TypeError, ValueError)):
            validate_distill_mode(bad)

    def test_extract_prompt_messages_strips_trailing_assistant(self):
        from soup_cli.utils.distill import extract_prompt_messages

        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        out = extract_prompt_messages(msgs)
        assert out == [{"role": "user", "content": "Q"}]
        # original not mutated
        assert len(msgs) == 2

    def test_extract_prompt_messages_no_trailing_assistant(self):
        from soup_cli.utils.distill import extract_prompt_messages

        msgs = [{"role": "user", "content": "Q"}]
        assert extract_prompt_messages(msgs) == msgs

    def test_build_sequence_distill_rows(self):
        from soup_cli.utils.distill import build_sequence_distill_rows

        class _FakeTok:
            eos_token = "</s>"

            def apply_chat_template(self, messages, tokenize, **kw):
                # return a tiny tensor-like via torch
                import torch

                return torch.tensor([[1, 2, 3]])

            def decode(self, ids, skip_special_tokens=True):
                return "teacher-says-hello"

        class _FakeTeacher:
            def generate(self, input_ids, max_new_tokens=None, **kw):
                import torch

                # echo prompt + 2 new tokens
                return torch.tensor([[1, 2, 3, 9, 10]])

            def to(self, *a, **k):
                return self

            @property
            def device(self):
                import torch

                return torch.device("cpu")

        rows = [{"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OLD"},
        ]}]
        out = build_sequence_distill_rows(
            rows, _FakeTeacher(), _FakeTok(), max_new_tokens=4, device="cpu"
        )
        assert len(out) == 1
        msgs = out[0]["messages"]
        assert msgs[0] == {"role": "user", "content": "Hi"}
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "teacher-says-hello"

    def test_schema_default_token(self):
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().distill_mode == "token"

    def test_schema_sequence_on_distill(self):
        cfg = load_config_from_string(
            "base: m\ntask: distill\ndata: {train: d.jsonl}\n"
            "training: {teacher_model: t, distill_mode: sequence}\n"
        )
        assert cfg.training.distill_mode == "sequence"

    def test_schema_distill_mode_outside_distill_rejected(self):
        _yaml_rejects(
            "base: m\ntask: sft\ndata: {train: d.jsonl}\n"
            "training: {distill_mode: sequence}\n",
            "distill",
        )

    def test_trainer_rejects_sequence_plus_uld_or_minillm(self):
        # The sequence+uld / sequence+minillm rejection lives in the distill
        # trainer's setup() (runtime — they are token/logit-level objectives),
        # not the config schema. Source-grep the guard so we catch a regression
        # without loading a model.
        import inspect

        from soup_cli.trainer import distill as distill_trainer

        src = inspect.getsource(distill_trainer)
        assert 'distill_mode", "token") == "sequence"' in src
        assert "uld_strategy is not None or tcfg.minillm_enabled" in src


# ---------------------------------------------------------------------------
# #146 — classifier LoRA path
# ---------------------------------------------------------------------------


class TestClassifierLora:
    def test_schema_default_false(self):
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().classifier_lora is False

    def test_schema_accept_on_classifier(self):
        cfg = load_config_from_string(
            "base: m\ntask: classifier\ndata: {train: d.jsonl}\n"
            "training: {num_labels: 3, classifier_lora: true}\n"
        )
        assert cfg.training.classifier_lora is True

    def test_schema_reject_outside_classifier_family(self):
        _yaml_rejects(
            "base: m\ntask: sft\ndata: {train: d.jsonl}\n"
            "training: {classifier_lora: true}\n",
            "classifier_lora",
        )

    def test_wrapper_sets_lora_active_attr(self):
        import inspect

        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        src = inspect.getsource(ClassifierTrainerWrapper)
        assert "_lora_active" in src
        assert "TaskType.SEQ_CLS" in src or "SEQ_CLS" in src


# ---------------------------------------------------------------------------
# #148 — LLaMA Pro per-arch zero-init handlers
# ---------------------------------------------------------------------------


def _make_fake_llama(num_layers=2, hidden=8, paths=(("mlp", "down_proj"), ("self_attn", "o_proj"))):
    """Build a fake Llama-shaped causal LM (nn.Module) for block-expansion tests."""
    from torch import nn

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            for outer, inner in paths:
                container = nn.Module()
                setattr(container, inner, nn.Linear(hidden, hidden))
                setattr(self, outer, container)

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(num_layers)])

    class _Cfg:
        num_hidden_layers = num_layers

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = _Cfg()

    return _Model()


class TestBlockExpansionPerArch:
    def test_arch_table_keys(self):
        from soup_cli.utils.block_expansion import _ARCH_RESIDUAL_PATHS

        for arch in (
            "LlamaForCausalLM", "MistralForCausalLM",
            "FalconForCausalLM", "GPT2LMHeadModel",
        ):
            assert arch in _ARCH_RESIDUAL_PATHS

    def test_arch_table_immutable(self):
        from soup_cli.utils.block_expansion import _ARCH_RESIDUAL_PATHS

        with pytest.raises(TypeError):
            _ARCH_RESIDUAL_PATHS["NewArch"] = ()  # type: ignore[index]

    def test_expand_appends_and_zero_inits(self):
        import torch

        from soup_cli.utils.block_expansion import expand_model_blocks

        model = _make_fake_llama(num_layers=2, hidden=8)
        # poison the residual projections so we can prove zero-init
        with torch.no_grad():
            for blk in model.model.layers:
                blk.mlp.down_proj.weight.fill_(7.0)
                blk.self_attn.o_proj.weight.fill_(7.0)
        # expand_model_blocks returns the TOTAL layer count after expansion.
        total = expand_model_blocks(model, 1)
        assert total == 3
        assert len(model.model.layers) == 3
        assert model.config.num_hidden_layers == 3
        new_blk = model.model.layers[-1]
        assert torch.count_nonzero(new_blk.mlp.down_proj.weight) == 0
        assert torch.count_nonzero(new_blk.self_attn.o_proj.weight) == 0

    def test_zero_returns_no_change(self):
        from soup_cli.utils.block_expansion import expand_model_blocks

        model = _make_fake_llama(num_layers=2)
        # no-op path returns the (unchanged) layer count, not 0.
        assert expand_model_blocks(model, 0) == 2
        assert len(model.model.layers) == 2

    def test_unknown_arch_warns(self, recwarn):
        from soup_cli.utils.block_expansion import expand_model_blocks

        # Falcon-shaped projections but reported as an unknown class name.
        model = _make_fake_llama(
            num_layers=1, hidden=8,
            paths=(("mlp", "weird_proj"), ("attn", "weird2")),
        )
        type(model).__name__ = "TotallyUnknownArch"
        expand_model_blocks(model, 1)
        msgs = " ".join(str(w.message) for w in recwarn.list)
        assert "residual" in msgs.lower() or "architecture" in msgs.lower()


# ---------------------------------------------------------------------------
# #158 — LongLoRA S² shift on Q/K projections
# ---------------------------------------------------------------------------


class TestLongLoRAShift:
    def test_shift_heads_for_s2_preserves_shape(self):
        import torch

        from soup_cli.utils.longlora import shift_heads_for_s2

        t = torch.randn(1, 4, 6, 8)  # [B, H, T, D]
        out = shift_heads_for_s2(t, group_size=4)
        assert out.shape == t.shape

    def test_shift_proj_block_shifts_second_half_heads(self):
        import torch

        from soup_cli.utils.longlora import _shift_proj_block

        # out: [B, T, n_heads*head_dim]
        b, t, n_heads, head_dim = 1, 6, 4, 3
        out = torch.arange(b * t * n_heads * head_dim, dtype=torch.float32).reshape(
            b, t, n_heads * head_dim
        )
        shifted = _shift_proj_block(out, head_dim=head_dim, n_heads=n_heads, group_size=4)
        assert shifted.shape == out.shape
        # first-half heads (0..n//2) along D stay put; second half rolled along T
        # → the tensor must differ overall
        assert not torch.equal(shifted, out)

    def test_separate_families(self):
        from soup_cli.utils.longlora import _FUSED_QKV_FAMILIES, _SEPARATE_QKV_FAMILIES

        assert "Llama" in _SEPARATE_QKV_FAMILIES
        assert "Mistral" in _SEPARATE_QKV_FAMILIES
        assert "Phi" in _FUSED_QKV_FAMILIES

    @staticmethod
    def _fake_llama_with_attention():
        """A model containing a registered ``LlamaAttention`` submodule.

        ``_install`` matches on the attention module's *class name* via the
        ``(?:Llama|Mistral|Qwen|Phi)\\w*Attention$`` regex, so the class must
        literally be named ``LlamaAttention`` (the model class name is
        irrelevant — the override walks every submodule).
        """
        from torch import nn

        class LlamaAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.config = types.SimpleNamespace(
                    num_attention_heads=2, num_key_value_heads=2, head_dim=4
                )

        attn = LlamaAttention()
        layer = nn.Module()
        layer.self_attn = attn
        inner = nn.Module()
        inner.layers = nn.ModuleList([layer])
        model = nn.Module()
        model.model = inner
        return model, attn

    def test_override_patches_qk_and_restores(self):
        import torch

        from soup_cli.utils.longlora import LongLoRAForwardOverride

        model, attn = self._fake_llama_with_attention()
        with LongLoRAForwardOverride(model, group_size=4):
            assert attn.q_proj.forward.__name__ == "s2_proj_shift"
            assert attn.k_proj.forward.__name__ == "s2_proj_shift"
            out = attn.q_proj(torch.randn(1, 6, 8))
            assert out.shape == (1, 6, 8)
        # restored on exit — patched wrapper gone (re-bound method, so compare
        # by name + the absence of the patch marker, not object identity).
        assert attn.q_proj.forward.__name__ == "forward"
        assert not getattr(attn.q_proj.forward, "_soup_longlora_patched", False)

    def test_override_idempotent(self):
        from soup_cli.utils.longlora import LongLoRAForwardOverride

        model, attn = self._fake_llama_with_attention()
        with LongLoRAForwardOverride(model, group_size=4):
            with LongLoRAForwardOverride(model, group_size=4):
                # nested install must not double-wrap
                assert attn.q_proj.forward.__name__ == "s2_proj_shift"


# ---------------------------------------------------------------------------
# #84 — Mixture-of-Depths live patch
# ---------------------------------------------------------------------------


class TestMixtureOfDepths:
    def test_validate_capacity_factor_happy(self):
        from soup_cli.utils.mod import validate_capacity_factor

        assert validate_capacity_factor(0.125) == pytest.approx(0.125)
        assert validate_capacity_factor(1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [True, "x", None, 0.0, -0.1, 1.5, float("nan"), float("inf")])
    def test_validate_capacity_factor_rejects(self, bad):
        from soup_cli.utils.mod import validate_capacity_factor

        with pytest.raises((TypeError, ValueError)):
            validate_capacity_factor(bad)

    def test_mod_capacity_floor_and_clamp(self):
        from soup_cli.utils.mod import mod_capacity

        assert mod_capacity(100, 0.125) == 12
        assert mod_capacity(4, 0.1) == 1  # floor clamps up to 1
        assert mod_capacity(10, 1.0) == 10

    def test_mod_capacity_rejects_bool_seq_len(self):
        from soup_cli.utils.mod import mod_capacity

        with pytest.raises(TypeError):
            mod_capacity(True, 0.5)

    @pytest.mark.parametrize("name,ok", [
        ("meta-llama/Llama-3.2-1B", True),
        ("Qwen/Qwen2.5-0.5B", True),
        ("mistralai/Mistral-7B", True),
        ("openai-community/gpt2", False),
        ("", False),
    ])
    def test_is_mod_supported_arch(self, name, ok):
        from soup_cli.utils.mod import is_mod_supported_arch

        assert is_mod_supported_arch(name) is ok

    def test_is_mod_supported_arch_nonstring(self):
        from soup_cli.utils.mod import is_mod_supported_arch

        assert is_mod_supported_arch(None) is False
        assert is_mod_supported_arch(123) is False

    def test_apply_mod_patch_adds_routers(self):
        from soup_cli.utils.mod import apply_mod_patch

        model = _make_fake_llama(num_layers=3, hidden=8)
        patched = apply_mod_patch(model, capacity_factor=0.5)
        assert patched == 3
        for layer in model.model.layers:
            assert getattr(layer, "_soup_mod_patched", False) is True
            assert hasattr(layer, "_soup_mod_router")
            assert layer.forward.__name__ == "mod_forward"

    def test_apply_mod_patch_idempotent(self):
        from soup_cli.utils.mod import apply_mod_patch

        model = _make_fake_llama(num_layers=2, hidden=8)
        assert apply_mod_patch(model, capacity_factor=0.5) == 2
        assert apply_mod_patch(model, capacity_factor=0.5) == 0  # already patched

    def test_apply_mod_if_configured_off(self):
        from soup_cli.utils.mod import apply_mod_if_configured

        tcfg = types.SimpleNamespace(use_mod=False, mod_capacity_factor=0.125)
        model = _make_fake_llama()
        assert apply_mod_if_configured(model, tcfg, "meta-llama/Llama-3.2-1B", None) == 0

    def test_apply_mod_if_configured_unsupported_warns(self, recwarn):
        from soup_cli.utils.mod import apply_mod_if_configured

        tcfg = types.SimpleNamespace(use_mod=True, mod_capacity_factor=0.125)
        model = _make_fake_llama()
        assert apply_mod_if_configured(model, tcfg, "openai-community/gpt2", None) == 0
        assert any("allowlist" in str(w.message).lower() for w in recwarn.list)

    def test_apply_mod_if_configured_supported(self):
        from soup_cli.utils.mod import apply_mod_if_configured

        tcfg = types.SimpleNamespace(use_mod=True, mod_capacity_factor=0.25)
        model = _make_fake_llama(num_layers=2, hidden=8)
        assert apply_mod_if_configured(model, tcfg, "meta-llama/Llama-3.2-1B", None) == 2

    def test_schema_use_mod_default_false(self):
        from soup_cli.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.use_mod is False
        assert tc.mod_capacity_factor == pytest.approx(0.125)

    def test_schema_use_mod_accept(self):
        cfg = load_config_from_string(
            "base: m\ntask: sft\ndata: {train: d.jsonl}\n"
            "training: {use_mod: true, mod_capacity_factor: 0.25}\n"
        )
        assert cfg.training.use_mod is True
        assert cfg.training.mod_capacity_factor == pytest.approx(0.25)

    def test_schema_mod_capacity_factor_bounds(self):
        _yaml_rejects(
            "base: m\ntask: sft\ndata: {train: d.jsonl}\n"
            "training: {mod_capacity_factor: 2.0}\n",
            "mod_capacity_factor",
        )


# ---------------------------------------------------------------------------
# #221 — live VeRA / VB-LoRA serving
# ---------------------------------------------------------------------------


def _make_bank(dim=8, users=("alice", "bob")):
    from soup_cli.utils.vector_bank import BankEntry, VectorBank

    return VectorBank(
        name="b",
        base_model="test/base",
        projection_seed=42,
        vector_dim=dim,
        entries=tuple(
            BankEntry(user_id=u, scaling=tuple([0.5 + 0.1 * i] * dim))
            for i, u in enumerate(users)
        ),
    )


class TestVectorBankServing:
    def test_reconstruct_projection_deterministic(self):
        import torch

        from soup_cli.utils.vector_bank import reconstruct_projection

        p1 = reconstruct_projection(42, 8)
        p2 = reconstruct_projection(42, 8)
        assert torch.allclose(p1, p2)
        assert p1.shape == (8, 8)

    def test_apply_bank_to_serve_returns_loaded(self):
        from soup_cli.utils.vector_bank import LoadedVectorBank, apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        assert isinstance(loaded, LoadedVectorBank)
        assert loaded.has_user("alice")
        assert not loaded.has_user("zoe")

    def test_set_active_user_contract(self):
        from soup_cli.utils.vector_bank import apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        assert loaded.set_active_user("alice") is True
        assert loaded.set_active_user("zoe") is False
        assert loaded._active_user is None  # unknown clears

    def test_delta_for_user_math(self):
        import torch

        from soup_cli.utils.vector_bank import apply_bank_to_serve, reconstruct_projection

        loaded = apply_bank_to_serve(_make_bank(dim=8))
        x = torch.ones(1, 3, 8)
        d = loaded.delta_for_user("alice", x)
        p = reconstruct_projection(42, 8)
        expected = 0.5 * (x @ p.transpose(0, 1))
        assert torch.allclose(d, expected, atol=1e-5)

    def test_delta_for_user_unknown_raises(self):
        from soup_cli.utils.vector_bank import apply_bank_to_serve

        loaded = apply_bank_to_serve(_make_bank())
        import torch

        with pytest.raises(KeyError):
            loaded.delta_for_user("zoe", torch.ones(1, 2, 8))

    def test_install_serve_hook_shifts_residual(self):
        import torch
        from torch import nn

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        # fake Llama-shaped model with a residual-producing layer
        class _Layer(nn.Module):
            def forward(self, x):
                return (x,)

        inner = nn.Module()
        inner.layers = nn.ModuleList([_Layer(), _Layer()])
        model = nn.Module()
        model.model = inner

        loaded = apply_bank_to_serve(_make_bank(dim=8))
        handle = loaded.install_serve_hook(model, layer=-1, strength=1.0)
        block = model.model.layers[-1]
        x = torch.ones(1, 3, 8)

        # no active user → no-op
        out_noop = block(x)[0]
        assert torch.allclose(out_noop, x)

        # active user → residual shifted by delta
        loaded.set_active_user("alice")
        out = block(x)[0]
        expected = x + loaded.delta_for_user("alice", x)
        assert torch.allclose(out, expected, atol=1e-5)
        handle.remove()

    def test_install_serve_hook_bad_layer(self):
        import torch  # noqa: F401
        from torch import nn

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity()])
        model = nn.Module()
        model.model = inner
        loaded = apply_bank_to_serve(_make_bank())
        with pytest.raises(ValueError, match="out of range"):
            loaded.install_serve_hook(model, layer=5)

    @pytest.mark.parametrize("bad", [True, "x", float("nan"), float("inf")])
    def test_install_serve_hook_bad_strength(self, bad):
        from torch import nn

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity()])
        model = nn.Module()
        model.model = inner
        loaded = apply_bank_to_serve(_make_bank())
        with pytest.raises((TypeError, ValueError)):
            loaded.install_serve_hook(model, strength=bad)

    @pytest.mark.parametrize("bad", [200.0, -200.0])
    def test_install_serve_hook_strength_cap(self, bad):
        # review fix — bank strength has a ±100 sanity cap (mirrors steering).
        from torch import nn

        from soup_cli.utils.vector_bank import apply_bank_to_serve

        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity()])
        model = nn.Module()
        model.model = inner
        loaded = apply_bank_to_serve(_make_bank())
        with pytest.raises(ValueError, match="sanity cap"):
            loaded.install_serve_hook(model, strength=bad)

    def test_load_bank_oversize_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.vector_bank import load_bank

        monkeypatch.chdir(tmp_path)
        big = tmp_path / "big.json"
        big.write_text("x" * (17 * 1024 * 1024), encoding="utf-8")
        with pytest.raises(ValueError, match="size"):
            load_bank("big.json")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink needs elevation on Windows")
    def test_load_bank_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.vector_bank import load_bank

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        os.symlink(real, link)
        with pytest.raises(ValueError):
            load_bank("link.json")

    def test_serve_flags_and_header(self):
        import inspect

        from soup_cli.commands import serve

        src = inspect.getsource(serve)
        assert "--bank" in src
        assert "X-User-Id" in src
        assert "set_active_user" in src
        assert "apply_bank_to_serve" in src


# ---------------------------------------------------------------------------
# #222 — live MoLE gating kernel + trainer
# ---------------------------------------------------------------------------


class TestMoleGatingKernel:
    def test_build_gating_kernel_softmax(self):
        import torch

        from soup_cli.utils.mole_routing import MoleGatingConfig, build_gating_kernel

        cfg = MoleGatingConfig(num_task_adapters=3, hidden_dim=8, temperature=1.0, top_k=3)
        kernel = build_gating_kernel(cfg)
        w = kernel(torch.randn(2, 4, 8))
        assert w.shape == (2, 4, 3)
        assert torch.allclose(w.sum(-1), torch.ones(2, 4), atol=1e-5)
        assert next(kernel.parameters()).requires_grad

    def test_build_gating_kernel_topk_sparse(self):
        import torch

        from soup_cli.utils.mole_routing import MoleGatingConfig, build_gating_kernel

        cfg = MoleGatingConfig(num_task_adapters=4, hidden_dim=8, temperature=1.0, top_k=2)
        kernel = build_gating_kernel(cfg)
        w = kernel(torch.randn(1, 5, 8))
        nonzero = (w > 1e-6).sum(-1)
        assert (nonzero <= 2).all()

    def test_build_gating_kernel_non_config(self):
        from soup_cli.utils.mole_routing import build_gating_kernel

        with pytest.raises(TypeError):
            build_gating_kernel({"num_task_adapters": 2})


class TestMoleTaskAdapters:
    def test_happy(self):
        from soup_cli.utils.mole_routing import validate_mole_task_adapters

        out = validate_mole_task_adapters(["./a", "./b", "./c"])
        assert out == ["./a", "./b", "./c"]

    @pytest.mark.parametrize("bad", [
        ["./a"],                 # < 2
        "not-a-list",            # non-list
        ["./a", "./a"],          # duplicate
        ["./a", ""],             # empty entry
        ["./a", "b\x00c"],       # null byte
        ["./a", 5],              # non-string
        ["./a", True],           # bool
        ["./a", "x" * 5000],     # oversize
    ])
    def test_rejects(self, bad):
        from soup_cli.utils.mole_routing import validate_mole_task_adapters

        with pytest.raises((TypeError, ValueError)):
            validate_mole_task_adapters(bad)

    def test_64_cap(self):
        from soup_cli.utils.mole_routing import validate_mole_task_adapters

        with pytest.raises(ValueError, match="cap"):
            validate_mole_task_adapters([f"./a{i}" for i in range(65)])


class TestMoleSchema:
    def test_happy(self):
        cfg = load_config_from_string(
            "base: m\ntask: moe_lora_routing\ndata: {train: d.jsonl}\n"
            "training: {mole_task_adapters: [./a, ./b, ./c], mole_top_k: 2, "
            "mole_temperature: 0.7}\n"
        )
        assert cfg.training.mole_task_adapters == ["./a", "./b", "./c"]
        assert cfg.training.mole_top_k == 2
        assert cfg.training.mole_temperature == pytest.approx(0.7)

    def test_fields_outside_task_rejected(self):
        _yaml_rejects(
            "base: m\ntask: sft\ndata: {train: d.jsonl}\n"
            "training: {mole_task_adapters: [./a, ./b]}\n",
            "moe_lora_routing",
        )

    def test_missing_adapters_rejected(self):
        _yaml_rejects(
            "base: m\ntask: moe_lora_routing\ndata: {train: d.jsonl}\ntraining: {}\n",
            "requires training.mole_task_adapters",
        )

    def test_top_k_exceeds_n_rejected(self):
        _yaml_rejects(
            "base: m\ntask: moe_lora_routing\ndata: {train: d.jsonl}\n"
            "training: {mole_task_adapters: [./a, ./b], mole_top_k: 5}\n",
            "exceeds",
        )

    def test_mlx_rejected(self):
        _yaml_rejects(
            "base: m\ntask: moe_lora_routing\nbackend: mlx\ndata: {train: d.jsonl}\n"
            "training: {mole_task_adapters: [./a, ./b]}\n",
            "mlx",
        )

    @pytest.mark.parametrize("bad_temp", [0.0, 200.0, True])
    def test_temperature_bounds(self, bad_temp):
        _yaml_rejects(
            "base: m\ntask: moe_lora_routing\ndata: {train: d.jsonl}\n"
            f"training: {{mole_task_adapters: [./a, ./b], mole_temperature: {bad_temp}}}\n",
            "mole_temperature",
        )


class TestMoleTrainer:
    def test_row_to_text(self):
        from soup_cli.trainer.mole_routing import _row_to_text

        assert _row_to_text({"text": "hi"}) == "hi"
        assert _row_to_text(
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        ) == "a\nb"
        assert _row_to_text({"prompt": "p", "completion": "c"}) == "pc"
        assert _row_to_text({}) == ""
        assert _row_to_text("not-a-dict") == ""

    def test_prepare_dataset(self):
        from soup_cli.trainer.mole_routing import _prepare_mole_dataset

        class _Tok:
            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [1, 2, 3]}

        rows = _prepare_mole_dataset([{"text": "hi"}, {}], _Tok(), 64)
        assert len(rows) == 1
        assert rows[0]["input_ids"] == [1, 2, 3]
        assert rows[0]["labels"] == [1, 2, 3]

    def test_make_trainer_class_factory(self):
        from soup_cli.trainer.mole_routing import make_mole_trainer_class

        class _Base:
            pass

        cls = make_mole_trainer_class(_Base)
        assert issubclass(cls, _Base)
        assert hasattr(cls, "compute_loss")
        # lru_cache returns the same subclass for the same base
        assert make_mole_trainer_class(_Base) is cls

    def test_train_before_setup_raises(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mole_routing import MoleRoutingTrainerWrapper

        cfg = load_config_from_string(
            "base: m\ntask: moe_lora_routing\ndata: {train: d.jsonl}\n"
            "training: {mole_task_adapters: [./a, ./b]}\n"
        )
        assert isinstance(cfg, SoupConfig)
        w = MoleRoutingTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="before setup"):
            w.train()


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(x) for x in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 12), soup_cli.__version__

    def test_train_routes_mole(self):
        import inspect

        from soup_cli.commands import train

        src = inspect.getsource(train)
        assert "moe_lora_routing" in src
        assert "MoleRoutingTrainerWrapper" in src

    @pytest.mark.parametrize("module", [
        "soup_cli.utils.mole_routing",
        "soup_cli.utils.mod",
    ])
    def test_no_top_level_torch_import(self, module):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "soup_cli", "utils", module.rsplit(".", 1)[-1] + ".py",
        )
        with open(src_path, encoding="utf-8") as fh:
            text = fh.read()
        assert "\nimport torch" not in text, f"{module} imports torch at top level"
        assert "\nfrom torch" not in text, f"{module} imports from torch at top level"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
