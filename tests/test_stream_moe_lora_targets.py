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
(A string target such as qwen4_exp's ``"all-linear"`` reaches the same helper on both
paths too; ``tests/test_moe_lora_string_targets.py`` covers it, streaming included.)

These tests drive the real streaming setup through the shared harness, with real
tiny models as the probe, and compare what it builds with what the resident
trainer sequence (``attach_preflight.trainer_lora_config``) builds for the same
config. The VRAM forecast is checked against a real peft attach.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from soup_cli.trainer.stream_setup import StreamingSetupMixin  # noqa: E402
from soup_cli.utils.moe import get_moe_target_modules  # noqa: E402
from soup_cli.utils.peft_wiring import resolve_lora_target_modules  # noqa: E402

_REAL_RESOLVER = resolve_lora_target_modules
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    """ANSI stripped and whitespace collapsed: Rich colours and wraps at 80 columns."""
    return " ".join(_ANSI_RE.sub("", text).split())


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


def _meta_deepseek_v3():
    """DeepSeek-V3's own dimensions (7168 hidden, 256 routed experts), two layers
    (one dense, one MoE), on the meta device: shapes without storage."""
    config = transformers.DeepseekV3Config(num_hidden_layers=2, first_k_dense_replace=1)
    with torch.device("meta"):
        return transformers.DeepseekV3ForCausalLM(config)


def _meta_qwen3_5_moe():
    """Qwen3.5-MoE's default dimensions (2048 hidden, 256 experts), four layers
    (three linear-attention, one full-attention), on the meta device."""
    config = transformers.Qwen3_5MoeTextConfig(num_hidden_layers=4)
    with torch.device("meta"):
        return transformers.Qwen3_5MoeForCausalLM(config)


#: label -> (what ``stream_arch_of`` reports for it, builder). Qwen3.5-MoE streams
#: as ``qwen3``. qwen3_moe, OLMoE and Mixtral are not here because ``stream_arch_of``
#: refuses them: streaming + moe_lora is reachable on deepseek_v3, Qwen3.5-MoE and
#: qwen4_exp only.
_BASES = {
    "deepseek_v3-q_lora_rank-int": ("deepseek_v3", lambda: _tiny_deepseek_v3(12)),
    "deepseek_v3-q_lora_rank-none": ("deepseek_v3", lambda: _tiny_deepseek_v3(None)),
    "qwen3_5_moe": ("qwen3", _tiny_qwen3_5_moe),
}


def _copy(targets):
    return list(targets) if isinstance(targets, list) else targets


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


def _streamed(
    tmp_path, monkeypatch, arch, model, lora, *, moe_lora=True, quantization="none", calls=None
):
    """The LoRA config kwargs the streaming setup hands ``build_streamed_model``.

    The probe the setup builds from ``cfg.base`` is the tiny real model, and its
    config is what ``AutoConfig`` returns, so the resolver and the MoE helper see a
    real architecture. Everything past them is the shared harness's stubs.
    ``calls``, when a dict, receives every call the harness recorded (the VRAM
    forecast's keyword arguments under "budget").
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
    calls = {} if calls is None else calls
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch=arch, model_cfg=model.config, load_calls=calls,
        tcfg_overrides={"lora": lora, "moe_lora": moe_lora, "quantization": quantization},
        resolve_targets=_REAL_RESOLVER,
    )
    assert len(calls["build"]) == 1
    return calls["build"][0]["lora_config"], probes


def _attach_streamed(monkeypatch, model, streamed):
    """The streamed LoRA kwargs, given to the real peft on the same model."""
    kwargs = dict(vars(streamed))
    monkeypatch.undo()  # the harness stubs peft and transformers; use the real ones
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(**kwargs))


def _adapted(attached):
    """Module keys carrying an adapter, in the base model's own naming."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    prefix = "base_model.model."
    return {
        name[len(prefix):] if name.startswith(prefix) else name
        for name, module in attached.named_modules()
        if isinstance(module, BaseTunerLayer)
    }


def _trainable(attached):
    return sum(param.numel() for param in attached.parameters() if param.requires_grad)


def _attention_projections(model):
    """Every ``nn.Linear`` directly under a ``self_attn`` block, read off the model."""
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-2:-1] == ["self_attn"]
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

        streamed, _probes = _streamed(tmp_path, monkeypatch, "qwen3", model, _lora("auto"))

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

    @pytest.mark.parametrize("base", sorted(_BASES))
    def test_a_name_both_lists_carry_appears_once(self, tmp_path, monkeypatch, base):
        """Expected value built from the raw MoE list, not from the helper, so a
        helper that stops de-duplicating fails here even though parity holds."""
        arch, build = _BASES[base]
        model = build()
        explicit = ["o_proj", "gate_proj", "my_custom_proj"]
        moe_names = get_moe_target_modules(model)
        assert {"o_proj", "gate_proj"} <= set(moe_names), moe_names

        streamed, _probes = _streamed(tmp_path, monkeypatch, arch, model, _lora(list(explicit)))

        targets = list(streamed.target_modules)
        assert len(targets) == len(set(targets)), targets
        assert targets == explicit + [name for name in moe_names if name not in explicit]

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

    @pytest.mark.parametrize(
        "targets", ["auto", ["o_proj", "my_custom_proj"]], ids=["auto", "explicit"]
    )
    def test_nf4_with_moe_lora_is_the_union_too(self, tmp_path, monkeypatch, targets):
        """The NF4 branch builds the probe whether or not moe_lora is set."""
        model = _tiny_deepseek_v3(12)
        resident = _resident(model, _lora(_copy(targets)))

        streamed, _probes = _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora(_copy(targets)),
            quantization="4bit",
        )

        assert list(streamed.target_modules) == list(resident)
        if isinstance(targets, list):
            assert list(streamed.target_modules)[: len(targets)] == targets


class TestTheUnionIsAnnounced:
    @pytest.mark.parametrize("quantization", ["none", "4bit"])
    def test_the_line_names_what_attaches(self, tmp_path, monkeypatch, capsys, quantization):
        """What the user reads before the run: the union's size and the fused
        expert parameters, printed by the helper when it is handed the console."""
        model = _tiny_deepseek_v3(12)

        streamed, _probes = _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"),
            quantization=quantization,
        )

        fused = list(streamed.target_parameters)
        expected = (
            f"ScatterMoE LoRA: targeting {len(streamed.target_modules)} module patterns "
            f"and {len(fused)} fused expert parameter(s): {', '.join(fused)}"
        )
        assert expected in _plain(capsys.readouterr().out)


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
    def test_deepseek_v3_every_attention_projection_and_the_experts(
        self, tmp_path, monkeypatch, q_lora_rank
    ):
        model = _tiny_deepseek_v3(q_lora_rank)
        expected = _attention_projections(model)
        assert expected, "the probe found no attention projection"

        streamed, _probes = _streamed(tmp_path, monkeypatch, "deepseek_v3", model, _lora("auto"))
        adapted = _adapted(_attach_streamed(monkeypatch, model, streamed))

        assert expected <= adapted, sorted(expected - adapted)
        assert any(".mlp.experts" in name for name in adapted), sorted(adapted)

    def test_qwen3_5_moe_both_attention_kinds_and_the_experts(self, tmp_path, monkeypatch):
        """A tiny real Qwen3.5-MoE builds on CPU in well under a second."""
        model = _tiny_qwen3_5_moe()
        expected = {
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.linear_attn.out_proj",
            *(
                f"model.layers.1.self_attn.{name}"
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            ),
        }

        streamed, _probes = _streamed(tmp_path, monkeypatch, "qwen3", model, _lora("auto"))
        adapted = _adapted(_attach_streamed(monkeypatch, model, streamed))

        assert expected <= adapted, sorted(expected - adapted)
        assert any(".mlp.experts" in name for name in adapted), sorted(adapted)


#: Bases the forecast is checked on. The tiny Qwen3.5-MoE is absent on purpose:
#: its attention keeps the default head sizes (q_proj 32 -> 1024), so the
#: estimator's hidden x hidden assumption is ~7x short there, before and after
#: this change. At Qwen3.5-MoE's own dimensions the routed experts dominate and the
#: forecast is high, which is the meta case below.
_FORECAST_BASES = {
    "tiny-deepseek_v3-q_lora_rank-int": ("deepseek_v3", lambda: _tiny_deepseek_v3(12)),
    "tiny-deepseek_v3-q_lora_rank-none": ("deepseek_v3", lambda: _tiny_deepseek_v3(None)),
    "deepseek_v3-real-dims-meta": ("deepseek_v3", _meta_deepseek_v3),
    "qwen3_5_moe-real-dims-meta": ("qwen3", _meta_qwen3_5_moe),
}


def _forecast(lora, model, targets, *, moe_lora=True):
    tcfg = SimpleNamespace(moe_lora=moe_lora, lora=lora)
    return StreamingSetupMixin._estimate_adapter_params(
        None, tcfg, model.config, target_modules=targets
    )


class TestTheVramForecastCountsWhatAttaches:
    """``_estimate_adapter_params`` documents one tolerance: it is "deliberately
    coarse and biased HIGH" (every target taken as hidden x hidden), never below
    what attaches, because the adapter term is ~0.5% of a streamed step's peak and
    an under-count eats the safety margin. No upper bound is documented; the 3x
    below is this file's own, to catch a runaway count. Measured: 2.1-2.4x on the
    tiny DeepSeek-V3, 1.80x at DeepSeek-V3's dimensions, 1.45x at Qwen3.5-MoE's."""

    def test_the_forecast_is_handed_the_attached_targets(self, tmp_path, monkeypatch):
        model = _tiny_deepseek_v3(12)
        calls = {}

        streamed, _probes = _streamed(
            tmp_path, monkeypatch, "deepseek_v3", model, _lora(["o_proj"]), calls=calls
        )

        assert len(calls["budget"]) == 1
        handed = calls["budget"][0]["target_modules"]
        assert list(handed) == list(streamed.target_modules)
        assert list(handed.target_parameters) == list(streamed.target_parameters)

    def test_the_budget_charges_the_targets_it_is_handed(self, monkeypatch):
        """The real ``_stream_budget_lines`` (the harness stubs it) passes the
        targets on: the peak forecast's adapter term is the union's count."""
        from soup_cli.utils import layer_stream

        model = _tiny_deepseek_v3(12)
        lora = _lora(["o_proj"])
        targets = _resident(model, lora)
        seen = {}
        real = layer_stream.estimate_stream_peak_vram

        def _capture(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", _capture)
        tcfg = SimpleNamespace(
            batch_size=1, stream_buffers=2, stream_vram_probe=False,
            stream_nf4_scale_cache=False, moe_lora=True, lora=lora,
        )
        StreamingSetupMixin()._stream_budget_lines(
            SimpleNamespace(data=SimpleNamespace(max_length=4)),
            tcfg,
            model_config=model.config,
            layer_bytes=1000,
            embed_bytes=0,
            index=SimpleNamespace(
                n_layers=2, quant_specs={}, dtype="float32", expert_layout="", expert_shapes={}
            ),
            on_cuda=False,
            target_modules=targets,
        )

        assert seen["adapter_params"] == _forecast(lora, model, targets)
        assert seen["adapter_params"] > _forecast(lora, model, None)

    @pytest.mark.parametrize("targets", ["auto", ["o_proj"]], ids=["auto", "explicit-o_proj"])
    @pytest.mark.parametrize("base", sorted(_FORECAST_BASES))
    def test_the_forecast_is_not_below_the_attached_adapter(
        self, tmp_path, monkeypatch, base, targets
    ):
        """``['o_proj']`` was charged ONE target per layer while ``moe_lora``
        attached the union plus every routed expert."""
        arch, build = _FORECAST_BASES[base]
        model = build()
        lora = _lora(_copy(targets))
        calls = {}

        streamed, _probes = _streamed(tmp_path, monkeypatch, arch, model, lora, calls=calls)
        forecast = _forecast(lora, model, calls["budget"][0]["target_modules"])
        trainable = _trainable(_attach_streamed(monkeypatch, model, streamed))

        assert trainable <= forecast <= 3 * trainable, (forecast, trainable)

    def test_a_list_holding_the_auto_names_is_charged_at_least_auto(self, tmp_path, monkeypatch):
        """A list that attaches everything ``auto`` does, and more, cannot be
        charged less. (A shorter list such as ``['o_proj']`` attaches less than
        ``auto`` -- measured 1600 vs 2256 parameters here -- so it is charged less.)"""
        model = _tiny_deepseek_v3(12)
        auto = list(_REAL_RESOLVER(model.config, "auto"))
        charged = {}
        for label, targets in (("auto", "auto"), ("superset", auto + ["my_custom_proj"])):
            root = tmp_path / label
            root.mkdir()
            lora = _lora(targets)
            calls = {}
            _streamed(root, monkeypatch, "deepseek_v3", model, lora, calls=calls)
            charged[label] = _forecast(lora, model, calls["budget"][0]["target_modules"])

        assert charged["superset"] >= charged["auto"], charged

    @pytest.mark.parametrize("targets", ["auto", ["o_proj"]], ids=["auto", "explicit-o_proj"])
    def test_without_moe_lora_the_forecast_is_unchanged(self, targets):
        """Only moe_lora's union is charged from the resolved list; without it the
        configured value is counted exactly as before."""
        model = _tiny_deepseek_v3(12)
        lora = _lora(_copy(targets))
        resolved = _REAL_RESOLVER(model.config, lora.target_modules)

        assert _forecast(lora, model, resolved, moe_lora=False) == _forecast(
            lora, model, None, moe_lora=False
        )


class TestTheForecastFormula:
    """The counts the docstring states, worked by hand. The ratio window above
    cannot tell the two expert rules apart on the test models: (E-1)*3 and E*2
    both land inside [1x, 3x]."""

    def test_fused_expert_tensors_are_one_adapter_per_routed_expert(self):
        """Under moe_lora: every listed module once per layer, plus E adapters per
        fused expert tensor; each adapter is hidden x hidden, A and B at rank r."""
        from soup_cli.utils.moe import MoeLoraTargets

        config = SimpleNamespace(hidden_size=32, num_hidden_layers=2, n_routed_experts=4)
        targets = MoeLoraTargets(
            ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        )
        tcfg = SimpleNamespace(moe_lora=True, lora=_lora("auto"))

        forecast = StreamingSetupMixin._estimate_adapter_params(
            None, tcfg, config, target_modules=targets
        )

        # 2 layers x (5 modules + 4 experts x 2 fused tensors) x 2 matrices x r=2 x 32
        assert forecast == 2 * (5 + 4 * 2) * 2 * 2 * 32 == 3328

    def test_a_dense_base_is_not_charged_differently_for_moe_lora(self):
        """No routed experts: moe_lora adds nothing. Without the ``experts > 1``
        guard an expert name would add (0 - 1) adapters, an UNDER-count."""
        config = SimpleNamespace(hidden_size=32, num_hidden_layers=2)
        lora = _lora(["q_proj", "gate_proj", "up_proj", "down_proj"])
        charged = {
            moe_lora: StreamingSetupMixin._estimate_adapter_params(
                None, SimpleNamespace(moe_lora=moe_lora, lora=lora), config
            )
            for moe_lora in (True, False)
        }

        assert charged[True] == charged[False] == 2 * 4 * 2 * 2 * 32
