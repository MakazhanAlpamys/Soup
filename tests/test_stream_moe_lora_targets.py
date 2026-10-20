"""Layer streaming + ``moe_lora`` keeps the attention targets, as resident training does.

The resident trainers resolve ``target_modules`` first and hand that list to
``resolve_moe_lora_targets``, which adds the MoE names and the fused expert
parameters to it (#1590). Layer streaming called the same helper with ``None`` --
it runs on a meta-skeleton probe that is deleted before the attach -- and then
REPLACED the resolved list with the helper's answer. So under ``stream_layers``:

- ``auto`` lost every name the MoE list does not repeat: Qwen3.5's linear-attention
  ``in_proj_qkv`` / ``out_proj``, and on DeepSeek-V3's MLA (which has no ``k_proj`` /
  ``v_proj``) ``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa`` and ``kv_b_proj``;
- an explicit ``lora.target_modules`` list was dropped altogether.

The replace was never a decision about streaming: #426 copied SFT's block from
before #1590, when ``auto`` still meant ``None`` and there was no list to keep.

These tests drive the real streaming setup through the shared harness, with real
tiny models as the probe, and compare what it builds with what the resident
trainer sequence (``attach_preflight.trainer_lora_config``) builds for the same
config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from soup_cli.utils.peft_wiring import resolve_lora_target_modules  # noqa: E402

_REAL_RESOLVER = resolve_lora_target_modules


def _tiny_deepseek_v3(q_lora_rank):
    config = transformers.DeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_position_embeddings=64,
    )
    return transformers.DeepseekV3ForCausalLM(config)


def _tiny_qwen3_5_moe():
    """One linear-attention and one full-attention layer, so both kinds exist."""
    config = transformers.Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=0,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=["linear_attention", "full_attention"],
    )
    return transformers.Qwen3_5MoeForCausalLM(config)


_BASES = {
    "deepseek_v3-q_lora_rank-int": ("deepseek_v3", lambda: _tiny_deepseek_v3(12)),
    "deepseek_v3-q_lora_rank-none": ("deepseek_v3", lambda: _tiny_deepseek_v3(None)),
    "qwen3_5_moe": ("qwen3_5_moe", _tiny_qwen3_5_moe),
}


def _lora(target_modules):
    """The LoRA block ``build_lora_config`` reads. Dropout 0: fused experts need it."""
    return SimpleNamespace(
        r=2, alpha=4, dropout=0.0, target_modules=target_modules, use_dora=False,
        use_rslora=False, rank_pattern=None, alpha_pattern=None,
    )


def _resident(model, lora, moe_lora=True):
    """What the resident trainers build: their own sequence, on the same model."""
    from soup_cli.utils.attach_preflight import trainer_lora_config

    cfg = SimpleNamespace(
        task="sft", modality="text", training=SimpleNamespace(moe_lora=moe_lora, lora=lora)
    )
    _peft_config, targets = trainer_lora_config(model, cfg)
    return targets


def _streamed(tmp_path, monkeypatch, arch, model, lora, *, moe_lora=True, quantization="none"):
    """The LoRA config kwargs the streaming setup hands ``build_streamed_model``.

    The probe the setup builds from ``cfg.base`` is the tiny real model, and its
    config is what ``AutoConfig`` returns, so the resolver and the MoE helper see a
    real architecture. Everything past them is the shared harness's stubs.
    """
    import soup_cli.trainer.stream_setup as stream_setup
    import soup_cli.utils.layer_stream_runtime as layer_stream_runtime
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    probes = []
    monkeypatch.setattr(
        stream_setup.StreamingSetupMixin, "_stream_meta_skeleton",
        classmethod(lambda cls, *a, **k: probes.append(k) or model),
    )
    monkeypatch.setattr(layer_stream_runtime, "quantised_layer_suffixes", lambda _probe: ())
    calls = {}
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch=arch, model_cfg=model.config, load_calls=calls,
        tcfg_overrides={"lora": lora, "moe_lora": moe_lora, "quantization": quantization},
        resolve_targets=_REAL_RESOLVER,
    )
    assert len(calls["build"]) == 1
    return calls["build"][0]["lora_config"], probes


def _attention_projections(model):
    """Every ``nn.Linear`` directly under an attention block, read off the model."""
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.split(".")[-2:-1] in (["self_attn"], ["linear_attn"])
    }


class TestTheAttentionTargetsAreKept:
    @pytest.mark.parametrize("base", sorted(_BASES))
    def test_auto_is_kept_and_the_moe_targets_are_added(self, tmp_path, monkeypatch, base):
        arch, build = _BASES[base]
        model = build()
        auto = _REAL_RESOLVER(model.config, "auto")
        resident = _resident(model, _lora("auto"))

        streamed, probes = _streamed(tmp_path, monkeypatch, arch, model, _lora("auto"))

        assert probes, "moe_lora did not build the probe"
        targets = list(streamed.target_modules)
        assert targets[: len(auto)] == list(auto), targets
        assert set(resident) - set(auto), "the MoE helper added nothing to compare"
        assert set(resident) <= set(targets)
        assert streamed.target_parameters, "the fused experts were not handed to peft"

    @pytest.mark.parametrize("base", sorted(_BASES))
    def test_the_streamed_targets_are_the_resident_ones(self, tmp_path, monkeypatch, base):
        """Parity with the resident trainer sequence: same module list, same order,
        same fused expert parameters."""
        arch, build = _BASES[base]
        model = build()
        resident = _resident(model, _lora("auto"))

        streamed, _probes = _streamed(tmp_path, monkeypatch, arch, model, _lora("auto"))

        assert list(streamed.target_modules) == list(resident)
        assert list(streamed.target_parameters) == list(resident.target_parameters)

    def test_qwen3_5_keeps_its_linear_attention_projections(self, tmp_path, monkeypatch):
        """The names the replace dropped on Qwen3.5-MoE."""
        model = _tiny_qwen3_5_moe()

        streamed, _probes = _streamed(tmp_path, monkeypatch, "qwen3_5_moe", model, _lora("auto"))

        assert {"in_proj_qkv", "out_proj"} <= set(streamed.target_modules)

    def test_deepseek_v3_keeps_its_mla_projections(self, tmp_path, monkeypatch):
        """The names the replace dropped on DeepSeek-V3: MLA has no k/v_proj, so
        q/k/v/o_proj alone left only ``o_proj`` adapted in attention."""
        model = _tiny_deepseek_v3(12)

        streamed, _probes = _streamed(tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"))

        mla = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"}
        assert mla <= set(streamed.target_modules)


class TestAnExplicitListSurvives:
    @pytest.mark.parametrize("base", sorted(_BASES))
    def test_the_user_list_comes_first_and_the_moe_targets_follow(
        self, tmp_path, monkeypatch, base
    ):
        arch, build = _BASES[base]
        model = build()
        explicit = ["o_proj", "my_custom_proj"]
        resident = _resident(model, _lora(list(explicit)))

        streamed, _probes = _streamed(tmp_path, monkeypatch, arch, model, _lora(list(explicit)))

        targets = list(streamed.target_modules)
        assert targets[:2] == explicit, targets
        assert targets == list(resident)

    def test_the_user_list_is_not_mutated(self, tmp_path, monkeypatch):
        model = _tiny_deepseek_v3(12)
        lora = _lora(["kv_b_proj"])

        _streamed(tmp_path, monkeypatch, "deepseek_v3", model, lora)

        assert lora.target_modules == ["kv_b_proj"]


class TestTheSameHelperGetsTheSameInput:
    """By construction: the helper is handed what the resolver returned, exactly as
    the resident trainers hand it, on both probe branches."""

    @pytest.mark.parametrize("quantization", ["none", "4bit"])
    def test_the_helper_receives_the_resolved_targets(
        self, tmp_path, monkeypatch, quantization
    ):
        import soup_cli.utils.moe as moe

        model = _tiny_deepseek_v3(12)
        seen = []
        real = moe.resolve_moe_lora_targets

        def _spy(probe, tcfg, target_modules, console=None):
            seen.append(target_modules)
            return real(probe, tcfg, target_modules, console=console)

        monkeypatch.setattr(moe, "resolve_moe_lora_targets", _spy)

        _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"),
            quantization=quantization,
        )

        assert seen == [_REAL_RESOLVER(model.config, "auto")]

    def test_nf4_with_moe_lora_is_the_union_too(self, tmp_path, monkeypatch):
        """The NF4 branch builds the probe whether or not moe_lora is set."""
        model = _tiny_deepseek_v3(12)
        resident = _resident(model, _lora("auto"))

        streamed, _probes = _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"), quantization="4bit"
        )

        assert list(streamed.target_modules) == list(resident)


class TestWithoutMoeLoraNothingChanges:
    @pytest.mark.parametrize("quantization", ["none", "4bit"])
    def test_the_resolved_list_reaches_the_lora_config(
        self, tmp_path, monkeypatch, quantization
    ):
        model = _tiny_deepseek_v3(12)

        streamed, _probes = _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"),
            moe_lora=False, quantization=quantization,
        )

        assert list(streamed.target_modules) == _REAL_RESOLVER(model.config, "auto")
        assert not streamed.target_parameters


class TestTheStreamedConfigAttaches:
    @pytest.mark.parametrize("q_lora_rank", [12, None], ids=["q_lora_rank-int", "q_lora_rank-none"])
    def test_every_attention_projection_and_the_experts_are_adapted(
        self, tmp_path, monkeypatch, q_lora_rank
    ):
        """The streamed kwargs, given to the real peft on the same model."""
        model = _tiny_deepseek_v3(q_lora_rank)
        expected = _attention_projections(model)
        assert expected, "the probe found no attention projection"

        streamed, _probes = _streamed(tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"))
        kwargs = dict(vars(streamed))
        monkeypatch.undo()  # the harness stubs peft and transformers; use the real ones

        from peft import LoraConfig, get_peft_model
        from peft.tuners.tuners_utils import BaseTunerLayer

        attached = get_peft_model(model, LoraConfig(**kwargs))
        prefix = "base_model.model."
        adapted = {
            name[len(prefix):] if name.startswith(prefix) else name
            for name, module in attached.named_modules()
            if isinstance(module, BaseTunerLayer)
        }

        assert expected <= adapted, sorted(expected - adapted)
        assert any(".mlp.experts" in name for name in adapted), sorted(adapted)
