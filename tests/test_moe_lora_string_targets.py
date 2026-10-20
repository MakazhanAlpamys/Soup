"""``moe_lora`` keeps a STRING ``target_modules`` and adds the MoE targets on top.

``resolve_moe_lora_targets`` merges the MoE names into a LIST target (#1590), but it
kept only lists: a string was replaced by the MoE list (``q/k/v/o`` and the expert
FFN names) plus the fused expert ``target_parameters``. peft gives a string two
meanings, and both were lost:

* ``"all-linear"``, peft's shorthand for every ``nn.Linear`` but the output head.
  ``target_modules: auto`` resolves to it on qwen4_exp, so ``moe_lora`` on a qwen4_exp
  base dropped the Gated DeltaNet ``in_proj_*`` / ``out_proj``, the QSA indexer, the
  shared-expert gate, the hyper-connection mixers and the PLE projections: 10 adapted
  modules instead of 34 on the tiny model below.
* any other string, which peft full-matches against every module name as a regex.

Now ``"all-linear"`` is kept as it is (it already reaches every module the MoE names
reach) with the fused expert parameters added, and a regex becomes the one regex
``(?:<user regex>)|(?:<the MoE names under peft's list rule>)``. Lists are unchanged.
"""

from __future__ import annotations

import functools
import re
import warnings
from io import StringIO
from types import SimpleNamespace

import pytest

from soup_cli.utils import moe as moe_mod
from soup_cli.utils.moe import (
    MoeLoraTargets,
    MoeLoraTargetString,
    resolve_moe_lora_targets,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: The names ``get_moe_target_modules`` returns for :class:`_FakeMoE`.
_MOE_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"]
_FUSED = ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
_NAMES_REGEX = r"(?:.*\.)?(?:q_proj|k_proj|v_proj|o_proj|down_proj|gate_proj|up_proj)"

#: Module names a regex is checked against: the MoE names, the qwen4_exp linears the
#: MoE list does not name, a router, the output head, and near-misses of both halves.
_KEYS = (
    "lm_head",
    "q_proj",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.Q_PROJ",
    "model.layers.0.self_attn.q_proj_extra",
    "model.layers.0.self_attnq_proj",
    "model.layers.0.mlp.gate",
    "model.layers.0.mlp.shared_expert.gate_proj",
    "model.layers.0.mlp.shared_expert_gate",
    "model.layers.0.linear_attn.in_proj_qkv",
    "model.layers.0.linear_attn.in_proj_z",
    "model.layers.0.linear_attn.out_proj",
    "model.layers.1.self_attn.indexer.index_qk_proj",
    "model.decoder.block.0.SelfAttention.q",
    "model.decoder.block.0.EncDecAttention.v",
    "model.encoder.block.0.SelfAttention.q",
)


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


class _FakeMoE:
    """Enough of a model for the helper: an MoE config, module names, 3-D experts."""

    def __init__(self, *, fused: bool = True):
        self.config = SimpleNamespace(model_type="qwen4_exp_text", num_experts=4)
        self._fused = fused

    def named_modules(self):
        names = (
            "",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.mlp.shared_expert.gate_proj",
            "model.layers.0.mlp.shared_expert.up_proj",
            "model.layers.0.mlp.shared_expert.down_proj",
            "lm_head",
        )
        return [(name, None) for name in names]

    def named_parameters(self):
        if not self._fused:
            return []
        return [
            (f"model.layers.0.{suffix}", SimpleNamespace(ndim=3)) for suffix in _FUSED
        ]


def _tcfg(target_modules="auto", *, moe_lora: bool = True, dropout: float = 0.0):
    return SimpleNamespace(
        moe_lora=moe_lora,
        moe_expert_quant=None,
        lora=SimpleNamespace(
            r=4,
            alpha=8,
            dropout=dropout,
            use_dora=False,
            use_rslora=False,
            use_vera=False,
            rank_pattern=None,
            alpha_pattern=None,
            init_strategy="random",
            target_modules=target_modules,
            target_parameters=None,
        ),
    )


def _resolve(target_modules, *, fused: bool = True, console=None, **tcfg):
    return resolve_moe_lora_targets(
        _FakeMoE(fused=fused), _tcfg(**tcfg), target_modules, console
    )


def _announced(target_modules, *, fused: bool = True) -> str:
    from rich.console import Console

    buf = StringIO()
    _resolve(target_modules, fused=fused, console=Console(file=buf, width=400))
    return _plain(buf.getvalue())


# --------------------------------------------------------------------------- #
# "all-linear"                                                                #
# --------------------------------------------------------------------------- #


class TestAllLinearIsKept:
    def test_all_linear_is_kept_and_the_fused_parameters_ride_along(self):
        targets = _resolve("all-linear")

        assert isinstance(targets, MoeLoraTargetString)
        assert targets == "all-linear"
        assert targets.target_parameters == _FUSED

    @pytest.mark.parametrize("spelling", ["all-linear", "ALL-LINEAR", "All-Linear"])
    def test_every_spelling_peft_accepts_is_kept_as_written(self, spelling):
        """peft lower-cases the string before comparing it with its shorthand."""
        targets = _resolve(spelling)

        assert targets == spelling
        assert "|" not in targets

    def test_the_shorthand_is_peft_s_own(self):
        pytest.importorskip("peft")
        from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND

        assert moe_mod.ALL_LINEAR == INCLUDE_LINEAR_LAYERS_SHORTHAND

    def test_a_per_expert_linear_checkpoint_keeps_it_with_no_parameters(self):
        targets = _resolve("all-linear", fused=False)

        assert targets == "all-linear"
        assert targets.target_parameters == []

    def test_build_lora_config_hands_peft_a_plain_string_and_the_parameters(self):
        pytest.importorskip("peft")
        from soup_cli.utils.peft_wiring import build_lora_config

        targets = _resolve("all-linear")
        config = build_lora_config(_tcfg().lora, target_modules=targets, task_type=None)

        assert type(config.target_modules) is str
        assert config.target_modules == "all-linear"
        assert sorted(config.target_parameters) == _FUSED


# --------------------------------------------------------------------------- #
# A regex                                                                     #
# --------------------------------------------------------------------------- #


def _peft_list_rule(names, key: str) -> bool:
    """How peft matches a LIST ``target_modules`` (``check_target_module_exists``)."""
    return key in names or any(key.endswith(f".{name}") for name in names)


class TestARegexBecomesOneRegexOfTheUnion:
    def test_the_documented_shape(self):
        targets = _resolve(r".*\.linear_attn\.in_proj_qkv")

        assert isinstance(targets, MoeLoraTargetString)
        assert targets == rf"(?:.*\.linear_attn\.in_proj_qkv)|(?:{_NAMES_REGEX})"
        assert targets.target_parameters == _FUSED

    @pytest.mark.parametrize(
        "user_regex",
        [
            r".*\.linear_attn\.in_proj_qkv",
            r".*\.linear_attn\.in_proj_(qkv|z)",
            r".*\.in_proj_z|.*\.out_proj",
            r".*\.[qkv]_proj",
            r"^model\.layers\.0\..*_proj$",
            r".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
            r"model\.layers\.(\d+)\.self_attn\.q_proj",
            r"(?i).*\.IN_PROJ_QKV",
            r"(?i)(?s).*\.IN_PROJ_Z",
            "(?x) .* \\. linear_attn \\. in_proj_qkv   # the Gated DeltaNet input",
            "in_proj_qkv",
        ],
    )
    def test_it_full_matches_exactly_what_either_half_matches(self, user_regex):
        """Every spelling peft would full-match: flags, anchors, groups, comments."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            joined = str(_resolve(user_regex))
            for key in _KEYS:
                expected = bool(re.fullmatch(user_regex, key)) or _peft_list_rule(
                    _MOE_NAMES, key
                )
                assert bool(re.fullmatch(joined, key)) is expected, key

    def test_a_leading_case_flag_stays_with_the_user_half(self):
        joined = str(_resolve(r"(?i).*\.IN_PROJ_QKV"))

        assert re.fullmatch(joined, "model.layers.0.linear_attn.in_proj_qkv")
        assert not re.fullmatch(joined, "model.layers.0.self_attn.Q_PROJ")
        assert re.fullmatch(joined, "model.layers.0.self_attn.q_proj")

    def test_a_trailing_verbose_comment_does_not_swallow_the_names(self):
        joined = str(_resolve("(?x) .* \\. in_proj_qkv  # input projection"))

        assert re.fullmatch(joined, "model.layers.0.self_attn.q_proj")
        assert re.fullmatch(joined, "model.layers.0.linear_attn.in_proj_qkv")

    def test_the_names_half_is_peft_s_list_rule_for_a_dotted_name_too(self):
        """A list entry may be a dotted suffix; its dot is a dot, not any character."""
        names = ["self_attn.q_proj", "w1"]
        names_regex = moe_mod._names_as_full_match_regex(names)
        for key in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attnXq_proj",
            "self_attn.q_proj",
            "model.layers.0.mlp.experts.0.w1",
            "model.layers.0.mlp.experts.0.xw1",
        ):
            assert bool(re.fullmatch(names_regex, key)) is _peft_list_rule(names, key), key

    def test_an_invalid_regex_is_refused_naming_the_field(self):
        with pytest.raises(ValueError, match=r"training\.lora\.target_modules is not a valid"):
            _resolve(r".*\.(q_proj")

    def test_peft_full_matches_the_joined_regex_as_the_union(self):
        """peft's own matcher, on peft's own configs, agrees key by key."""
        pytest.importorskip("peft")
        from peft import LoraConfig
        from peft.tuners.tuners_utils import check_target_module_exists

        user_regex = r".*\.linear_attn\.in_proj_(qkv|z)"
        joined = LoraConfig(target_modules=str(_resolve(user_regex)))
        alone = LoraConfig(target_modules=user_regex)
        names = LoraConfig(target_modules=list(_MOE_NAMES))
        for key in _KEYS:
            expected = bool(check_target_module_exists(alone, key)) or bool(
                check_target_module_exists(names, key)
            )
            assert bool(check_target_module_exists(joined, key)) is expected, key


# --------------------------------------------------------------------------- #
# What did not change                                                         #
# --------------------------------------------------------------------------- #


class TestWhatDidNotChange:
    @pytest.mark.parametrize("target_modules", ["all-linear", r".*\.in_proj_qkv"])
    def test_without_moe_lora_a_string_is_returned_untouched(self, target_modules):
        assert _resolve(target_modules, moe_lora=False) is target_modules

    def test_a_dense_model_returns_a_string_untouched(self):
        dense = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        assert resolve_moe_lora_targets(dense, _tcfg(), "all-linear") == "all-linear"

    def test_a_list_is_extended_as_before(self):
        targets = _resolve(["in_proj_qkv", "q_proj"])

        assert isinstance(targets, MoeLoraTargets)
        assert list(targets) == ["in_proj_qkv", "q_proj"] + [
            name for name in _MOE_NAMES if name != "q_proj"
        ]
        assert targets.target_parameters == _FUSED

    @pytest.mark.parametrize("unresolved", [None, "auto"])
    def test_no_targets_and_the_unresolved_sentinel_get_the_moe_list(self, unresolved):
        """``auto`` is Soup's placeholder, not a peft regex: it means "decide for me"."""
        targets = _resolve(unresolved)

        assert isinstance(targets, MoeLoraTargets)
        assert list(targets) == _MOE_NAMES

    def test_an_unmapped_auto_gets_the_moe_list(self):
        from soup_cli.utils.peft_wiring import UnmappedTargets

        targets = _resolve(UnmappedTargets(["qwen2_moe"]))

        assert isinstance(targets, MoeLoraTargets)
        assert list(targets) == _MOE_NAMES


# --------------------------------------------------------------------------- #
# The announcement                                                            #
# --------------------------------------------------------------------------- #


class TestTheAnnouncementSaysWhatAttaches:
    def test_a_list_line_is_unchanged(self):
        assert _announced(["in_proj_qkv"]) == (
            "ScatterMoE LoRA: targeting 8 module patterns and 2 fused expert "
            "parameter(s): mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )

    def test_all_linear_says_it_is_kept_and_what_is_added(self):
        assert _announced("all-linear") == (
            "ScatterMoE LoRA: keeping target_modules 'all-linear' (it already covers "
            "the 7 MoE module patterns) and adding 2 fused expert parameter(s): "
            "mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )

    def test_all_linear_without_fused_experts_adds_nothing(self):
        assert _announced("all-linear", fused=False) == (
            "ScatterMoE LoRA: keeping target_modules 'all-linear' (it already covers "
            "the 7 MoE module patterns)"
        )

    def test_a_regex_says_it_was_joined_with_the_names(self):
        assert _announced(r".*\.in_proj_qkv") == (
            "ScatterMoE LoRA: targeting the target_modules regex or the 7 MoE module "
            "patterns (one combined regex) and 2 fused expert parameter(s): "
            "mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )

    def test_a_regex_without_fused_experts(self):
        assert _announced(r".*\.in_proj_qkv", fused=False) == (
            "ScatterMoE LoRA: targeting the target_modules regex or the 7 MoE module "
            "patterns (one combined regex)"
        )

    def test_the_user_regex_is_not_printed_as_markup(self):
        """A character class is a Rich tag; the line never carries the user's text."""
        line = _announced(r".*\.[qkv]_proj|.*\.[/]")

        assert "[qkv]" not in line
        assert line.startswith("ScatterMoE LoRA: targeting the target_modules regex")


# --------------------------------------------------------------------------- #
# The dropout refusal reads a string as a string                              #
# --------------------------------------------------------------------------- #


class TestTheDropoutRefusalReadsAStringAsAString:
    def test_dropout_with_fused_experts_is_refused_for_a_string(self):
        with pytest.raises(ValueError, match=r"training\.lora\.dropout: 0\.0"):
            _resolve("all-linear", dropout=0.05)

    def test_all_linear_on_a_family_peft_converts_is_refused(self):
        """No fused tensors, but peft rewrites ``all-linear`` into target_parameters.

        On a ``mixtral``-type model peft's v5 conversion adds the router and the
        expert names to an ``all-linear`` target and routes them to
        ``lora.ParamWrapper``, which refuses dropout. The probe used to read the
        string as ``list("all-linear")``, one character per target.
        """
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model

        model = _per_expert_linear_moe(torch)
        assert moe_mod.find_fused_expert_parameters(model) == []
        assert moe_mod.peft_routes_lora_to_fused_params(model, "all-linear") is True
        with pytest.raises(ValueError, match=r"training\.lora\.dropout: 0\.0"):
            resolve_moe_lora_targets(model, _tcfg(dropout=0.05), "all-linear")
        # ...and peft really does refuse it: the refusal is not an over-refusal.
        with pytest.raises(ValueError, match="ParamWrapper"):
            get_peft_model(
                _per_expert_linear_moe(torch),
                LoraConfig(r=4, target_modules="all-linear", lora_dropout=0.05),
            )

    def test_a_regex_peft_keeps_as_modules_keeps_dropout(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")

        model = _per_expert_linear_moe(torch)
        targets = resolve_moe_lora_targets(
            model, _tcfg(dropout=0.05), r".*\.self_attn\.v_proj"
        )

        assert re.fullmatch(str(targets), "layers.0.self_attn.v_proj")


def _per_expert_linear_moe(torch):
    """The transformers-4 layout, one ``nn.Linear`` per expert, as a ``mixtral``."""

    class _PerExpertLinearMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="mixtral", num_local_experts=2)
            experts = torch.nn.ModuleList(
                torch.nn.ModuleDict(
                    {
                        "gate_proj": torch.nn.Linear(8, 16),
                        "up_proj": torch.nn.Linear(8, 16),
                        "down_proj": torch.nn.Linear(16, 8),
                    }
                )
                for _ in range(2)
            )
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "self_attn": torch.nn.ModuleDict(
                                {name: torch.nn.Linear(8, 8) for name in ("q_proj", "v_proj")}
                            ),
                            "block_sparse_moe": torch.nn.ModuleDict(
                                {"gate": torch.nn.Linear(8, 2), "experts": experts}
                            ),
                        }
                    )
                ]
            )

    return _PerExpertLinearMoE()


# --------------------------------------------------------------------------- #
# End to end: tiny real models, the trainer's sequence, real get_peft_model    #
# --------------------------------------------------------------------------- #

_SMALL = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    pad_token_id=0,
)


def _build(arch: str):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    torch.manual_seed(0)
    if arch == "qwen4_exp":
        try:
            from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig
        except ImportError:
            pytest.skip("installed Transformers release does not include qwen4_exp")
        text = Qwen4ExpTextConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts_per_tok=1,
            num_experts=2,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=64,
            hc_count=2,
            hc_lowrank=4,
            ple_layer_ids=[1],
            use_cache=False,
            indexer_n_heads=1,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=8,
            indexer_compress_ratio=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        return transformers.AutoModelForCausalLM.from_config(
            Qwen4ExpConfig(text_config=text.to_dict())
        )
    if arch == "qwen3_moe":
        return transformers.Qwen3MoeForCausalLM(
            transformers.Qwen3MoeConfig(
                **_SMALL, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16
            )
        )
    if arch == "deepseek_v3":
        return transformers.DeepseekV3ForCausalLM(
            transformers.DeepseekV3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                n_routed_experts=4,
                num_experts_per_tok=2,
                n_shared_experts=1,
                n_group=1,
                topk_group=1,
                first_k_dense_replace=1,
                q_lora_rank=16,
                kv_lora_rank=16,
                qk_rope_head_dim=8,
                qk_nope_head_dim=8,
                v_head_dim=8,
                pad_token_id=0,
            )
        )
    raise KeyError(arch)


@functools.lru_cache(maxsize=None)
def _attach(arch: str, target_modules, *, moe_lora: bool):
    """``attach_preflight.trainer_lora_config`` (the trainer's sequence), then peft.

    Cached: each test reads the same few attaches, and a result is frozen sets.
    """
    from peft import get_peft_model

    from soup_cli.utils.attach_preflight import trainer_lora_config

    model = _build(arch)
    cfg = SimpleNamespace(
        task="sft", modality="text", training=_tcfg(target_modules, moe_lora=moe_lora)
    )
    peft_config, targets = trainer_lora_config(model, cfg)
    tuner = get_peft_model(model, peft_config).base_model
    return (
        frozenset(tuner.targeted_module_names),
        frozenset(tuner.targeted_parameter_names),
        targets,
    )


@functools.lru_cache(maxsize=None)
def _routed_expert_parameters(arch: str) -> frozenset[str]:
    return frozenset(
        name
        for name, param in _build(arch).named_parameters()
        if param.ndim == 3 and ".experts." in name
    )


@functools.lru_cache(maxsize=None)
def _routers(arch: str) -> frozenset[str]:
    return frozenset(
        name
        for name, module in _build(arch).named_modules()
        if name.endswith(".mlp.gate") and "router" in type(module).__name__.lower()
    )


#: (arch, target_modules). ``auto`` on qwen4_exp is the reported case: it resolves to
#: ``"all-linear"``. On qwen3_moe and deepseek_v3 peft's own v5 conversion rewrites a
#: string target, and the union still holds.
_STRING_CASES = [
    ("qwen4_exp", "auto"),
    ("qwen3_moe", "all-linear"),
    ("deepseek_v3", "all-linear"),
    ("qwen4_exp", r".*\.linear_attn\.in_proj_(qkv|z)"),
    ("qwen3_moe", r"model\.layers\.0\.self_attn\.q_proj"),
    ("deepseek_v3", r".*\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)"),
]


class TestTheUnionAttachesOnTinyRealModels:
    @pytest.mark.parametrize(("arch", "target_modules"), _STRING_CASES)
    def test_the_adapted_set_is_the_string_s_plus_the_moe_targets(self, arch, target_modules):
        string_modules, string_params, _ = _attach(arch, target_modules, moe_lora=False)
        moe_modules, moe_params, _ = _attach(arch, None, moe_lora=True)
        modules, params, _ = _attach(arch, target_modules, moe_lora=True)

        assert modules == string_modules | moe_modules
        assert params == string_params | moe_params

    @pytest.mark.parametrize(("arch", "target_modules"), _STRING_CASES)
    def test_the_routed_experts_are_adapted(self, arch, target_modules):
        _, params, _ = _attach(arch, target_modules, moe_lora=True)
        experts = _routed_expert_parameters(arch)

        assert experts
        assert experts <= params

    @pytest.mark.parametrize(("arch", "target_modules"), _STRING_CASES)
    def test_nothing_outside_the_intended_set(self, arch, target_modules):
        """No output head; a router only where peft's own reading of the string has it.

        peft's ``all-linear`` leaves out the output embedding everywhere. On qwen4_exp
        the router is a parameter-only module, outside ``all-linear``. On the families
        peft converts from the v4 layout (qwen3_moe, deepseek_v3) its ``all-linear``
        adds the router weight itself, as v4's did when the router was an
        ``nn.Linear``: ``moe_lora`` neither adds it nor takes it away.
        """
        string_modules, string_params, _ = _attach(arch, target_modules, moe_lora=False)
        modules, params, _ = _attach(arch, target_modules, moe_lora=True)
        routers = _routers(arch)
        router_weights = {f"{name}.weight" for name in routers}

        assert routers
        assert not any(name.endswith("lm_head") for name in modules)
        assert not routers & modules
        assert router_weights & params == router_weights & string_params
        if arch == "qwen4_exp" or target_modules != "all-linear":
            assert not router_weights & params

    def test_qwen4_exp_gets_back_every_linear_the_moe_list_dropped(self):
        modules, _, targets = _attach("qwen4_exp", "auto", moe_lora=True)
        moe_modules, _, _ = _attach("qwen4_exp", None, moe_lora=True)
        leaves = {name.rsplit(".", 1)[-1] for name in modules}

        assert targets == "all-linear"
        assert len(modules) == 34 and len(moe_modules) == 10
        assert {
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "index_qk_proj",
            "shared_expert_gate",
            "input_mix_weight_down",
            "input_mix_weight_up",
            "block_inject_weight",
            "key_proj",
            "value_proj",
        } <= leaves

    def test_the_regex_reaches_peft_as_one_plain_string(self):
        from soup_cli.utils.attach_preflight import trainer_lora_config

        user_regex = r".*\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)"
        cfg = SimpleNamespace(
            task="sft", modality="text", training=_tcfg(user_regex, moe_lora=True)
        )
        peft_config, _ = trainer_lora_config(_build("deepseek_v3"), cfg)

        assert type(peft_config.target_modules) is str
        assert peft_config.target_modules == rf"(?:{user_regex})|(?:{_NAMES_REGEX})"
        assert sorted(peft_config.target_parameters) == _FUSED
