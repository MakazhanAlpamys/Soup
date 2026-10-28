"""``target_modules: auto`` on DeepSeek-V3 MLA follows the base's ``q_lora_rank``.

``DeepseekV3Attention`` (transformers 5.17, ``modeling_deepseek_v3.py``) builds ONE
of two query paths, and the checkpoint's config picks which:

    q_lora_rank: None    ->  q_proj
    q_lora_rank: <int>   ->  q_a_proj -> q_a_layernorm -> q_b_proj

The ``deepseek_v3`` entry of ``MOE_TEXT_LORA_TARGETS`` named only the low-rank pair.
peft refuses an attach only when NO target matches, so on a base with
``q_lora_rank: null`` (Moonlight-16B-A3B) the kv/o names matched, the attach
succeeded, and the query path trained without an adapter. Nothing said so.

The rule is applied where the attention IS that class and its config is the one
the row is chosen by: ``deepseek_v3`` and ``kimi_k2`` (Kimi-K2 checkpoints declare
``DeepseekV3ForCausalLM``). Not on ``glm_moe_dsa``: its config refuses
``q_lora_rank: None`` outright. Not on ``kimi_k25`` either, which keeps main's
low-rank tuple whatever ``text_config.q_lora_rank`` says: its vision tower has a
``q_proj`` of its own, so a row would have to be a language-tower regex, and
``resolve_moe_lora_targets`` (``utils/moe.py``) keeps the resolved targets only when
they are a list -- under ``moe_lora: true`` that regex cost the language tower its
``kv_a_proj_with_mqa`` / ``kv_b_proj`` adapters. No shipped Kimi K2.5 / K2.6 config
sets ``q_lora_rank: null``.

The resolver half uses config doubles; the end-to-end half builds tiny real models
on CPU and checks the adapter against the module tree the model actually has.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.utils.peft_wiring import (
    MLA_FULL_RANK_QUERY_TARGETS,
    MOE_TEXT_LORA_TARGETS,
    resolve_lora_target_modules,
)

#: What DeepseekV3Attention defines with ``q_lora_rank: None``, attention only.
FULL_RANK_QUERY = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
#: ... and with an integer ``q_lora_rank`` (the table entry since #1070).
LOW_RANK_QUERY = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]

_ABSENT = object()


def _config(model_type, q_lora_rank=_ABSENT, text_config=None):
    """A config double; ``_ABSENT`` leaves the field off entirely."""
    config = SimpleNamespace(model_type=model_type, text_config=text_config)
    if q_lora_rank is not _ABSENT:
        config.q_lora_rank = q_lora_rank
    return config


def _model(model_type, q_lora_rank=_ABSENT, text_config=None):
    return SimpleNamespace(config=_config(model_type, q_lora_rank, text_config))


def _kimi_k25(q_lora_rank=_ABSENT):
    """The VL wrapper: the MLA fields live on the text config only."""
    return _model("kimi_k25", text_config=_config("deepseek_v3", q_lora_rank))


class TestTheResolverReadsQLoraRank:
    @pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
    def test_none_targets_the_full_rank_q_proj(self, model_type):
        """Moonlight's shape. The low-rank pair does not exist on this model."""
        resolved = resolve_lora_target_modules(_model(model_type, None), "auto")

        assert resolved == FULL_RANK_QUERY

    @pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
    def test_an_int_keeps_the_low_rank_pair(self, model_type):
        """DeepSeek-V3 and Kimi-K2 themselves (1536): unchanged by this fix."""
        resolved = resolve_lora_target_modules(_model(model_type, 1536), "auto")

        assert resolved == LOW_RANK_QUERY
        assert resolved == list(MOE_TEXT_LORA_TARGETS[model_type])

    @pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
    def test_zero_keeps_the_low_rank_pair(self, model_type):
        """transformers tests ``q_lora_rank is None``, so 0 builds ``q_a_proj`` /
        ``q_b_proj`` (pinned on the real class below). A falsy test would swap
        them for a ``q_proj`` this model does not have."""
        resolved = resolve_lora_target_modules(_model(model_type, 0), "auto")

        assert resolved == LOW_RANK_QUERY

    @pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2", "glm_moe_dsa"])
    def test_a_config_without_the_field_keeps_the_table_entry(self, model_type):
        """A missing field is not ``None``. Guessing a layout from it would swap
        a list that matches for one that may not."""
        resolved = resolve_lora_target_modules(_model(model_type), "auto")

        assert resolved == list(MOE_TEXT_LORA_TARGETS[model_type])

    def test_glm_moe_dsa_keeps_its_entry_even_if_a_double_says_none(self):
        """GLM-5's config cannot say ``None`` (pinned below); if a double does,
        the resolver must not start guessing a layout transformers will not build."""
        resolved = resolve_lora_target_modules(_model("glm_moe_dsa", None), "auto")

        assert resolved == list(MOE_TEXT_LORA_TARGETS["glm_moe_dsa"])

    def test_a_bare_config_is_read_like_a_model(self):
        """Layer streaming hands the resolver the ``AutoConfig`` itself, with no
        ``.config`` around it (``stream_setup.py``). The field must be seen there."""
        resolved = resolve_lora_target_modules(_config("deepseek_v3", None), "auto")

        assert resolved == FULL_RANK_QUERY

    def test_an_explicit_list_is_untouched(self):
        resolved = resolve_lora_target_modules(_model("deepseek_v3", None), ["q_a_proj"])

        assert resolved == ["q_a_proj"]

    def test_a_returned_list_is_a_copy(self):
        first = resolve_lora_target_modules(_model("deepseek_v3", None), "auto")
        first.append("mutated")

        assert "mutated" not in resolve_lora_target_modules(_model("deepseek_v3", None), "auto")


class TestTheKimiWrapperKeepsTheLowRankTuple:
    """``kimi_k25`` resolves as on main whatever its text config says (see the
    module docstring for why it has no full-rank row)."""

    @pytest.mark.parametrize(
        "q_lora_rank", [None, 1536, _ABSENT], ids=["none", "shipped-int", "absent"]
    )
    def test_resolves_exactly_what_main_resolves(self, q_lora_rank):
        """``LOW_RANK_QUERY`` is main's ``kimi_k25`` entry, written out rather than
        read from the table. The first version of this fix, a regex row read
        through ``text_config``, failed the ``none`` case."""
        resolved = resolve_lora_target_modules(_kimi_k25(q_lora_rank), "auto")

        assert resolved == LOW_RANK_QUERY

    def test_the_full_rank_rows_are_the_text_only_mla_types(self):
        """The rule reads only the config whose ``model_type`` chose the row, so a
        ``kimi_k25`` row (its MLA fields sit on ``text_config``) would be dead
        here, and live it would need the regex ``moe_lora`` discards."""
        assert set(MLA_FULL_RANK_QUERY_TARGETS) == {"deepseek_v3", "kimi_k2"}

    def test_an_unrecognised_wrapper_keeps_the_table_entry(self):
        """Reached through the text config of a wrapper Soup has no row for. The
        table's MLA names exist only in the text tower; a bare ``q_proj`` does
        not (Kimi's vision tower has one), so the q_lora_rank rule stays off."""
        model = _model("some_unseen_wrapper", text_config=_config("deepseek_v3", None))

        resolved = resolve_lora_target_modules(model, "auto")

        assert resolved == list(MOE_TEXT_LORA_TARGETS["deepseek_v3"])

    def test_a_wrapper_saying_none_itself_still_keeps_the_table_entry(self):
        """The rule is off in the text-config fallback even when the outer config
        carries ``q_lora_rank: None`` too: a wrapper Soup has no row for gets the
        table entry whatever either config says."""
        model = _model(
            "some_unseen_wrapper", None, text_config=_config("deepseek_v3", None)
        )

        resolved = resolve_lora_target_modules(model, "auto")

        assert resolved == list(MOE_TEXT_LORA_TARGETS["deepseek_v3"])


# --------------------------------------------------------------------------
# tiny real models on CPU
# --------------------------------------------------------------------------
def _tiny_v3_fields(q_lora_rank):
    return dict(
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


def _tiny_deepseek_v3(q_lora_rank):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.DeepseekV3Config(**_tiny_v3_fields(q_lora_rank))
    return transformers.DeepseekV3ForCausalLM(config)


def _soup_cfg(**training):
    """A real SoupConfig. ``target_modules`` is left to the schema default."""
    import yaml

    from soup_cli.config.loader import load_config_from_string

    raw = {
        "base": "org/m",
        "task": "sft",
        "data": {"train": "./x.jsonl", "format": "alpaca"},
        "training": {"lora": {"r": 4, "alpha": 8, "dropout": 0.0}, **training},
    }
    return load_config_from_string(yaml.safe_dump(raw))


def _attention_projections(model):
    """Every ``nn.Linear`` directly under a ``self_attn``, read off the model."""
    import torch

    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-2:-1] == ["self_attn"]
    }


def _attach(model, cfg):
    """The trainer's own target sequence (``trainer_lora_config``), then peft."""
    pytest.importorskip("peft")
    from peft import get_peft_model

    from soup_cli.utils.attach_preflight import trainer_lora_config

    peft_config, _targets = trainer_lora_config(model, cfg)
    return get_peft_model(model, peft_config)


def _adapted(peft_model):
    """Module keys carrying a LoRA adapter, in the base model's own naming."""
    from peft.tuners.lora import LoraLayer

    prefix = "base_model.model."
    return {
        name[len(prefix):] if name.startswith(prefix) else name
        for name, module in peft_model.named_modules()
        if isinstance(module, LoraLayer)
    }


class TestEveryAttentionProjectionIsAdapted:
    def test_the_config_default_is_auto(self):
        """The end-to-end tests below rely on the default, not on a value they set."""
        assert _soup_cfg().training.lora.target_modules == "auto"

    @pytest.mark.parametrize(
        ("q_lora_rank", "query"),
        [(None, ["q_proj"]), (12, ["q_a_proj", "q_b_proj"])],
        ids=["q_lora_rank-none", "q_lora_rank-int"],
    )
    def test_deepseek_v3(self, q_lora_rank, query):
        model = _tiny_deepseek_v3(q_lora_rank)
        expected = _attention_projections(model)
        assert expected, "the probe found no attention projection"

        adapted = _adapted(_attach(model, _soup_cfg()))

        assert adapted == expected
        for layer in range(2):
            for name in query:
                assert f"model.layers.{layer}.self_attn.{name}" in adapted

    def test_deepseek_v3_with_moe_lora_keeps_every_attention_projection(self):
        """Plain training merges ``moe_lora``'s targets (``q/k/v/o_proj`` + the
        expert FFN names) INTO the ``auto`` list, so ``kv_a_proj_with_mqa`` /
        ``kv_b_proj`` stay adapted under the flag. A replace would drop them; layer
        streaming does replace, which is left alone here (see the PR)."""
        model = _tiny_deepseek_v3(None)
        expected = _attention_projections(model)

        adapted = _adapted(_attach(model, _soup_cfg(moe_lora=True)))

        assert expected <= adapted, sorted(expected - adapted)

    @pytest.mark.filterwarnings("ignore:Initializing zero-element tensors")
    def test_a_zero_q_lora_rank_builds_the_low_rank_pair_and_keeps_it(self):
        """The premise of ``test_zero_keeps_the_low_rank_pair``, on the real class:
        ``DeepseekV3Attention`` tests ``is None``, so 0 is the low-rank layout."""
        model = _tiny_deepseek_v3(0)
        names = {name.rsplit(".", 1)[-1] for name in _attention_projections(model)}

        assert names == set(LOW_RANK_QUERY)
        assert resolve_lora_target_modules(model, "auto") == LOW_RANK_QUERY

    @pytest.mark.parametrize("q_lora_rank", [None, 12], ids=["q_lora_rank-none", "q_lora_rank-int"])
    def test_kimi_k25_adapts_main_s_names_in_the_language_tower_only(self, q_lora_rank):
        """Main's low-rank tuple either way. On an integer that is every attention
        projection the language tower has; on ``null`` its ``q_proj`` is left out,
        the documented limit (no shipped K2.5 / K2.6 config is ``null``)."""
        model = _tiny_kimi_k25(q_lora_rank)
        language = _kimi_language_attention(model)
        vision_q = {
            name for name, _ in model.named_modules()
            if "vision_tower" in name and name.endswith(".q_proj")
        }
        assert language and vision_q, "the probe found nothing to compare"

        adapted = _adapted(_attach(model, _soup_cfg()))

        assert adapted == {name for name in language if _leaf(name) in LOW_RANK_QUERY}
        assert not any("vision_tower" in name for name in adapted)
        if q_lora_rank is not None:
            assert adapted == language

    def test_kimi_k25_with_moe_lora_and_a_null_rank_keeps_the_kv_adapters(self):
        """The review finding: a language-tower regex row is dropped by
        ``resolve_moe_lora_targets``, which keeps the resolved targets only when
        they are a list, so ``kv_a_proj_with_mqa`` / ``kv_b_proj`` lost the adapter
        main gives them under ``moe_lora: true``."""
        model = _tiny_kimi_k25(None)
        kv = {
            name for name in _kimi_language_attention(model)
            if _leaf(name) in ("kv_a_proj_with_mqa", "kv_b_proj")
        }
        assert kv, "the probe found no kv projection"

        adapted = _adapted(_attach(model, _soup_cfg(moe_lora=True)))

        assert kv <= adapted, sorted(kv - adapted)


def _leaf(name):
    return name.rsplit(".", 1)[-1]


def _tiny_kimi_k25(q_lora_rank):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Kimi_K25ForConditionalGeneration"):
        pytest.skip("this transformers has no native Kimi K2.5")

    text = dict(model_type="deepseek_v3", **_tiny_v3_fields(q_lora_rank))
    vision = dict(
        patch_size=2, pos_emb_height=4, pos_emb_width=4, pos_emb_time=1,
        num_attention_heads=2, num_hidden_layers=1, hidden_size=16, intermediate_size=32,
    )
    config = transformers.Kimi_K25Config(
        text_config=text, vision_config=vision, projection_hidden_size=16
    )
    return transformers.Kimi_K25ForConditionalGeneration(config)


def _kimi_language_attention(model):
    return {name for name in _attention_projections(model) if ".language_model." in name}


class TestWhyGlmMoeDsaHasNoFullRankEntry:
    def test_its_config_refuses_a_null_q_lora_rank(self):
        """GLM-5's attention init has the same if/else as DeepSeek-V3's, but its
        DSA indexer and its forward use ``q_a_proj`` unconditionally, and the
        config refuses ``None`` before either runs. If a transformers release
        lets it build, revisit the ``glm_moe_dsa`` row."""
        pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "GlmMoeDsaForCausalLM"):
            pytest.skip("this transformers has no GLM-MoE-DSA")

        # Broad on purpose: a strict-dataclass refusal naming the field, or, on a
        # release without the strict typing, the indexer's nn.Linear(None, ...).
        with pytest.raises(Exception) as excinfo:
            transformers.GlmMoeDsaForCausalLM(
                transformers.GlmMoeDsaConfig(**_tiny_v3_fields(None))
            )

        assert "q_lora_rank" in str(excinfo.value) or isinstance(excinfo.value, TypeError)


class TestLayerStreamingHandsOverTheSameTargets:
    """``stream_setup`` resolves from the ``AutoConfig``, not from a model."""

    @pytest.mark.parametrize(
        ("q_lora_rank", "expected"),
        [(None, FULL_RANK_QUERY), (12, LOW_RANK_QUERY)],
        ids=["q_lora_rank-none", "q_lora_rank-int"],
    )
    def test_the_lora_config_built_for_the_streamed_model(
        self, tmp_path, monkeypatch, q_lora_rank, expected
    ):
        transformers = pytest.importorskip("transformers")
        from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

        real_resolver = resolve_lora_target_modules
        model_cfg = transformers.DeepseekV3Config(**_tiny_v3_fields(q_lora_rank))
        lora = SimpleNamespace(
            r=2, alpha=4, dropout=0.0, target_modules="auto", use_dora=False,
            use_rslora=False, rank_pattern=None, alpha_pattern=None,
        )
        calls = {}

        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=model_cfg,
            load_calls=calls, tcfg_overrides={"lora": lora}, resolve_targets=real_resolver,
        )

        assert len(calls["build"]) == 1
        assert calls["build"][0]["lora_config"].target_modules == expected
