"""``moe_lora`` keeps a STRING ``target_modules`` and adds the MoE targets on top.

``resolve_moe_lora_targets`` merges the MoE names into a LIST target (#1590), but it
kept only lists: a string was replaced by the MoE list (``q/k/v/o`` and the expert
FFN names) plus the fused expert ``target_parameters``. peft gives a string two
meanings, and both were lost:

* ``"all-linear"``, peft's shorthand for every ``nn.Linear`` but the output head.
  ``target_modules: auto`` resolves to it on qwen4_exp, so ``moe_lora`` on a qwen4_exp
  base dropped the Gated DeltaNet ``in_proj_*`` / ``out_proj``, the QSA indexer, the
  hyper-connection mixers and the PLE projections. On the tiny model below that is 10
  adapted modules under ``moe_lora`` before this fix and 32 after it: the 34 linears
  ``all-linear`` adapts without ``moe_lora``, minus the 2 ``shared_expert_gate``,
  whose output is a routing logit.
* any other string, which peft full-matches against every module name as a regex.

Now ``"all-linear"`` is kept as it is (it already reaches every module the MoE names
reach) with the fused expert parameters added, and a regex becomes the one regex
``(?:<user regex>)|(?:<the MoE names under peft's list rule>)``. Lists are unchanged.

No MoE router or shared-expert gate gets an adapter from a string under ``moe_lora``
(a module whose output is routing logits): they go to peft's ``exclude_modules``, or,
on the families where peft's v4->v5 conversion would still adapt the router weight as
a parameter (qwen3_moe, deepseek_v3, mixtral, ...), the string becomes an explicit
list of what it means without them. Without ``moe_lora`` a string reaches peft as
before, router included where peft includes it.
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


def _tcfg(
    target_modules="auto", *, moe_lora: bool = True, dropout: float = 0.0, use_vera=False
):
    return SimpleNamespace(
        moe_lora=moe_lora,
        moe_expert_quant=None,
        lora=SimpleNamespace(
            r=4,
            alpha=8,
            dropout=dropout,
            use_dora=False,
            use_rslora=False,
            use_vera=use_vera,
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
# Stand-ins: the transformers-4 layout, one nn.Linear per expert               #
# --------------------------------------------------------------------------- #


def _per_expert_linear_moe(
    torch,
    *,
    model_type="mixtral",
    shared_gate=False,
    attn_gate=False,
    dense_out=False,
    layers=1,
    experts=2,
    block_gate_proj=None,
):
    """One ``nn.Linear`` per expert and an ``nn.Linear`` router, as ``model_type``.

    peft converts a ``mixtral`` from this layout (its ``all-linear`` there adds the
    router weight); a ``qwen2_moe`` it does not. ``shared_gate`` adds a one-logit
    ``shared_expert_gate``; ``attn_gate`` a non-router ``nn.Linear`` whose leaf name is
    the router's, ``gate``; ``dense_out`` an attention ``nn.Linear`` no MoE name
    reaches; ``block_gate_proj`` (an output size) a ``gate_proj`` beside the router,
    whose path starts with the router's.
    """

    class _PerExpertLinearMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type=model_type, num_local_experts=experts)
            blocks = []
            for _ in range(layers):
                attention = {name: torch.nn.Linear(8, 8) for name in ("q_proj", "v_proj")}
                if attn_gate:
                    attention["gate"] = torch.nn.Linear(8, 8)
                if dense_out:
                    attention["dense_out"] = torch.nn.Linear(8, 8)
                per_expert = torch.nn.ModuleList(
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": torch.nn.Linear(8, 16),
                            "up_proj": torch.nn.Linear(8, 16),
                            "down_proj": torch.nn.Linear(16, 8),
                        }
                    )
                    for _ in range(experts)
                )
                block = {"gate": torch.nn.Linear(8, experts), "experts": per_expert}
                if block_gate_proj is not None:
                    block["gate_proj"] = torch.nn.Linear(8, block_gate_proj)
                if shared_gate:
                    block["shared_expert_gate"] = torch.nn.Linear(8, 1)
                blocks.append(
                    torch.nn.ModuleDict(
                        {
                            "self_attn": torch.nn.ModuleDict(attention),
                            "block_sparse_moe": torch.nn.ModuleDict(block),
                        }
                    )
                )
            self.layers = torch.nn.ModuleList(blocks)

    return _PerExpertLinearMoE()


def _jetmoe_like(torch):
    """JetMoE's layout without its experts child, plus the near-misses it must ignore.

    ``mlp.router`` wraps one ``nn.Linear`` beside a 3-D expert tensor; ``mlp.gate_proj``
    is a dense projection with the same output size; ``attn.gate`` is an attention
    gate with no expert tensor beside it. ``mixture`` is JetMoE's attention layout: an
    ``experts`` module that is not a per-expert list (three different children, so
    not three experts), whose router sits inside it, beside a ``kv_proj`` with three
    outputs.
    """

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(8, 4, bias=False)

    class _ParallelExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 16, 8))

    class _JetMoeLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="jetmoe", num_local_experts=4)
            self.mlp = torch.nn.ModuleDict(
                {
                    "input_linear": _ParallelExperts(),
                    "router": _Router(),
                    "gate_proj": torch.nn.Linear(8, 4),
                }
            )
            self.attn = torch.nn.ModuleDict(
                {"gate": torch.nn.Linear(8, 4), "q_proj": torch.nn.Linear(8, 8)}
            )
            mixture = torch.nn.ModuleDict(
                {
                    "input_linear": _ParallelExperts(),
                    "output_linear": _ParallelExperts(),
                    "router": _Router(),
                }
            )
            self.mixture = torch.nn.ModuleDict(
                {"experts": mixture, "kv_proj": torch.nn.Linear(8, 3)}
            )

    return _JetMoeLike()


def _peft_attach(model, peft_config):
    from peft import get_peft_model

    tuner = get_peft_model(model, peft_config).base_model
    return set(tuner.targeted_module_names), set(tuner.targeted_parameter_names)


def _stand_in_attach(tcfg, target_modules, **layout):
    """Resolve on one stand-in, attach the config ``build_lora_config`` makes to another."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from soup_cli.utils.peft_wiring import build_lora_config

    targets = resolve_moe_lora_targets(
        _per_expert_linear_moe(torch, **layout), tcfg, target_modules
    )
    peft_config = build_lora_config(tcfg.lora, target_modules=targets, task_type=None)
    modules, params = _peft_attach(_per_expert_linear_moe(torch, **layout), peft_config)
    return targets, peft_config, modules, params


_ROUTER = "layers.0.block_sparse_moe.gate"
_SHARED_GATE = "layers.0.block_sparse_moe.shared_expert_gate"


class TestTheDropoutRefusalReadsAStringAsAString:
    def test_dropout_with_fused_experts_is_refused_for_a_string(self):
        with pytest.raises(ValueError, match=r"training\.lora\.dropout: 0\.0"):
            _resolve("all-linear", dropout=0.05)

    def test_the_probe_reads_a_string_as_a_string(self):
        """peft rewrites ``all-linear`` into target_parameters on a ``mixtral``-type
        model (the router and the expert names); ``list()`` of the string would be one
        target per character, which peft rewrites into nothing."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")

        model = _per_expert_linear_moe(torch)
        assert moe_mod.find_fused_expert_parameters(model) == []
        assert moe_mod.peft_routes_lora_to_fused_params(model, "all-linear") is True

    def test_without_the_router_all_linear_keeps_dropout_on_that_family(self):
        """The router was the one target peft wraps in ``lora.ParamWrapper`` there."""
        torch = pytest.importorskip("torch")
        targets, _, modules, params = _stand_in_attach(_tcfg(dropout=0.05), "all-linear")
        from peft import LoraConfig, get_peft_model

        assert isinstance(targets, MoeLoraTargets)
        assert _ROUTER not in modules and not params
        # ...where peft's own all-linear, router included, refuses the dropout.
        with pytest.raises(ValueError, match="ParamWrapper"):
            get_peft_model(
                _per_expert_linear_moe(torch),
                LoraConfig(r=4, target_modules="all-linear", lora_dropout=0.05),
            )

    def test_a_regex_on_that_family_stays_a_string_and_keeps_dropout(self):
        targets, peft_config, modules, params = _stand_in_attach(
            _tcfg(dropout=0.05), r".*\.self_attn\.v_proj"
        )

        assert isinstance(targets, MoeLoraTargetString)
        assert not peft_config.exclude_modules
        assert "layers.0.self_attn.v_proj" in modules
        assert _ROUTER not in modules and not params


class TestTheGatesStayOutOnStandIns:
    def test_a_shared_expert_gate_is_left_out_of_the_list(self):
        targets, _, modules, params = _stand_in_attach(_tcfg(), "all-linear", shared_gate=True)

        assert isinstance(targets, MoeLoraTargets)
        assert "shared_expert_gate" not in targets
        assert not {_ROUTER, _SHARED_GATE} & modules and not params
        assert "layers.0.block_sparse_moe.experts.1.down_proj" in modules

    def test_a_non_router_named_gate_is_dropped_rather_than_reach_the_routers(self):
        """peft's conversion rewrites any entry ending ``.gate`` into ``gate.weight``,
        which lands on every router: such an entry cannot be kept without them."""
        targets, _, modules, params = _stand_in_attach(_tcfg(), "all-linear", attn_gate=True)

        assert not [entry for entry in targets if entry.rsplit(".", 1)[-1] == "gate"]
        assert _ROUTER not in modules and not params
        assert "layers.0.self_attn.q_proj" in modules

    def test_a_regex_reaching_a_non_router_named_gate_becomes_a_list(self):
        """Kept, the regex would make peft rewrite ``self_attn.gate`` onto the router."""
        targets, _, modules, params = _stand_in_attach(
            _tcfg(), r".*\.self_attn\.(q_proj|gate)", attn_gate=True
        )

        assert isinstance(targets, MoeLoraTargets)
        assert "q_proj" in targets
        assert not [entry for entry in targets if entry.rsplit(".", 1)[-1] == "gate"]
        assert _ROUTER not in modules and not params

    def test_exclude_modules_keeps_linear_gates_out_where_peft_does_not_convert(self):
        targets, peft_config, modules, params = _stand_in_attach(
            _tcfg(), "all-linear", model_type="qwen2_moe", shared_gate=True
        )

        assert isinstance(targets, MoeLoraTargetString) and targets == "all-linear"
        assert set(peft_config.exclude_modules) == {_ROUTER, _SHARED_GATE}
        assert not {_ROUTER, _SHARED_GATE} & modules and not params
        assert "layers.0.block_sparse_moe.experts.0.gate_proj" in modules

    @pytest.mark.parametrize("model_type", ["qwen2_moe", "mixtral"])
    def test_a_gate_proj_beside_the_router_is_still_adapted(self, model_type):
        """``…block_sparse_moe.gate_proj`` starts with the router's path, not inside it."""
        gate_proj = "layers.0.block_sparse_moe.gate_proj"
        targets, peft_config, modules, params = _stand_in_attach(
            _tcfg(), "all-linear", model_type=model_type, block_gate_proj=16
        )

        assert gate_proj in modules
        assert _ROUTER not in modules and not params
        if model_type == "qwen2_moe":
            assert isinstance(targets, MoeLoraTargetString)
            assert list(peft_config.exclude_modules) == [_ROUTER]

    def test_vera_gets_a_list_because_vera_has_no_exclude_modules(self):
        targets, peft_config, modules, _ = _stand_in_attach(
            _tcfg(use_vera=True), "all-linear", model_type="qwen2_moe", shared_gate=True
        )

        assert isinstance(targets, MoeLoraTargets)
        assert type(peft_config).__name__ == "VeraConfig"
        assert not {_ROUTER, _SHARED_GATE} & modules
        assert "layers.0.self_attn.q_proj" in modules

    def test_vera_keeps_a_regex_that_reaches_no_gate(self):
        targets, peft_config, modules, _ = _stand_in_attach(
            _tcfg(use_vera=True), r".*\.self_attn\.q_proj", model_type="qwen2_moe",
            shared_gate=True,
        )

        assert isinstance(targets, MoeLoraTargetString)
        assert type(peft_config.target_modules) is str
        assert not {_ROUTER, _SHARED_GATE} & modules
        assert "layers.0.self_attn.q_proj" in modules

    def test_a_listed_regex_is_what_it_full_matches_not_a_prefix(self):
        """VeRA lists this regex (it reaches the router); ``dense`` is only a prefix."""
        targets, _, modules, _ = _stand_in_attach(
            _tcfg(use_vera=True),
            r"layers\.0\.self_attn\.dense|.*\.block_sparse_moe\.gate",
            model_type="qwen2_moe",
            dense_out=True,
        )

        assert isinstance(targets, MoeLoraTargets)
        assert not [entry for entry in targets if entry.endswith("dense_out")]
        assert "layers.0.self_attn.dense_out" not in modules and _ROUTER not in modules

    def test_a_leaf_that_reaches_beyond_the_regex_is_listed_by_path(self):
        targets, _, modules, _ = _stand_in_attach(
            _tcfg(use_vera=True),
            r"layers\.0\.self_attn\.dense_out|.*\.block_sparse_moe\.gate",
            model_type="qwen2_moe",
            dense_out=True,
            layers=2,
        )

        assert "layers.0.self_attn.dense_out" in targets
        assert "dense_out" not in targets
        assert "layers.0.self_attn.dense_out" in modules
        assert "layers.1.self_attn.dense_out" not in modules

    def test_the_list_is_announced_with_its_count(self):
        from rich.console import Console

        torch = pytest.importorskip("torch")
        pytest.importorskip("peft")
        buf = StringIO()
        resolve_moe_lora_targets(
            _per_expert_linear_moe(torch, attn_gate=True),
            _tcfg(),
            r".*\.self_attn\.(q_proj|gate)",
            Console(file=buf, width=400),
        )

        assert _plain(buf.getvalue()) == (
            "ScatterMoE LoRA: targeting the target_modules regex as a list of 7 target "
            "names: the modules it matches, without the 1 MoE router/gate module(s), plus "
            "the 7 MoE module patterns"
        )


class TestFailClosedWithoutPeftsPrivateHelpers:
    """No answer from peft means a list, never a string peft might route to a router."""

    def test_without_the_all_linear_expander_the_string_becomes_a_list(self, monkeypatch):
        monkeypatch.setattr(moe_mod, "_peft_all_linear_expander", lambda: None)
        modules, params, targets = _attach_uncached("qwen4_exp", "auto")
        expected_modules, expected_params, _ = _expected("qwen4_exp", "auto")

        assert isinstance(targets, MoeLoraTargets)
        assert modules == expected_modules and params == expected_params

    def test_an_all_linear_expander_that_raises_fails_closed(self, monkeypatch):
        """A private helper whose signature changed raises; setup lists, not crashes."""

        def changed_signature(_model):
            raise TypeError("_maybe_include_all_linear_layers() missing 1 required argument")

        monkeypatch.setattr(moe_mod, "_peft_all_linear_expander", lambda: changed_signature)
        modules, params, targets = _attach_uncached("qwen4_exp", "auto")
        expected_modules, expected_params, gates = _expected("qwen4_exp", "auto")

        assert isinstance(targets, MoeLoraTargets)
        assert modules == expected_modules and params == expected_params
        assert not gates & modules

    def test_without_the_conversion_the_string_becomes_a_list(self, monkeypatch):
        monkeypatch.setattr(moe_mod, "_peft_config_converter", lambda: None)
        modules, params, targets = _attach_uncached("qwen4_exp", "auto")
        expected_modules, expected_params, _ = _expected("qwen4_exp", "auto")

        assert isinstance(targets, MoeLoraTargets)
        assert modules == expected_modules and params == expected_params

    def test_a_peft_without_the_private_modules_fails_closed(self, monkeypatch):
        """The import guards: a peft that lacks either private module answers nothing."""
        import sys

        from soup_cli.utils.peft_wiring import build_lora_config

        model = _read_only("qwen4_exp")
        monkeypatch.setitem(sys.modules, "peft.tuners.tuners_utils", None)
        monkeypatch.setitem(sys.modules, "peft.utils.transformers_weight_conversion", None)

        assert moe_mod._peft_all_linear_expander() is None
        assert moe_mod._peft_config_converter() is None
        targets = resolve_moe_lora_targets(model, _tcfg(), "all-linear")
        monkeypatch.undo()
        modules, params = _peft_attach(
            _build("qwen4_exp"),
            build_lora_config(_tcfg().lora, target_modules=targets, task_type=None),
        )
        expected_modules, expected_params, _ = _expected("qwen4_exp", "auto")

        assert isinstance(targets, MoeLoraTargets)
        assert modules == expected_modules and params == expected_params

    def test_without_the_conversion_a_gate_named_entry_is_dropped(self, monkeypatch):
        monkeypatch.setattr(moe_mod, "_peft_config_converter", lambda: None)
        targets, _, modules, params = _stand_in_attach(_tcfg(), "all-linear", attn_gate=True)

        assert not [entry for entry in targets if entry.rsplit(".", 1)[-1] == "gate"]
        assert _ROUTER not in modules and not params


class TestARegexThatMatchesNothingIsAnnounced:
    def test_a_bare_name_written_as_a_string_is_warned_about(self):
        line = _announced("in_proj_qkv")

        assert line.startswith(
            "training.lora.target_modules is a regex that matches no module of this model:"
        )
        assert "only moe_lora's own targets attach" in line

    def test_a_regex_that_matches_says_nothing_extra(self):
        assert not _announced(r".*\.in_proj_qkv").startswith("training.lora")


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
    if arch == "dbrx":
        return transformers.DbrxForCausalLM(
            transformers.DbrxConfig(
                d_model=32,
                n_heads=2,
                n_layers=2,
                max_seq_len=64,
                vocab_size=64,
                attn_config={"kv_n_heads": 1, "rope_theta": 10000.0},
                ffn_config={"ffn_hidden_size": 16, "moe_num_experts": 4, "moe_top_k": 2},
            )
        )
    if arch == "jetmoe":
        return transformers.JetMoeForCausalLM(
            transformers.JetMoeConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=2,
                num_key_value_heads=2,
                kv_channels=8,
                intermediate_size=16,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
            )
        )
    if arch == "dense":
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**_SMALL))
    raise KeyError(arch)


@functools.lru_cache(maxsize=None)
def _read_only(arch: str):
    """One model per architecture for the checks that only read it."""
    return _build(arch)


def _attach_uncached(arch: str, target_modules, *, moe_lora: bool = True):
    """``attach_preflight.trainer_lora_config`` (the trainer's sequence), then peft."""
    from soup_cli.utils.attach_preflight import trainer_lora_config

    model = _build(arch)
    cfg = SimpleNamespace(
        task="sft", modality="text", training=_tcfg(target_modules, moe_lora=moe_lora)
    )
    peft_config, targets = trainer_lora_config(model, cfg)
    modules, params = _peft_attach(model, peft_config)
    return frozenset(modules), frozenset(params), targets


@functools.lru_cache(maxsize=None)
def _attach(arch: str, target_modules, *, moe_lora: bool):
    """Cached: each test reads the same few attaches, and a result is frozen sets."""
    return _attach_uncached(arch, target_modules, moe_lora=moe_lora)


#: The families among these that peft converts from the transformers-4 layout. There
#: peft resolves a regex to the last names of the modules it matches, and renames a
#: target whose last name is a v4 expert projection into the fused expert parameter,
#: so an ``nn.Linear`` with that leaf name (deepseek_v3's dense MLP and its shared
#: experts) is not adapted as a module, whatever the target: peft's behaviour, the
#: same with a list.
_PEFT_CONVERTS = {"qwen3_moe", "deepseek_v3"}
_V4_EXPERT_NAMES = {"gate_proj", "up_proj", "down_proj"}


@functools.lru_cache(maxsize=None)
def _expected(arch: str, target_modules):
    """What should attach, worked out from the model alone.

    The string's meaning without the gates (each ``*Router`` module and each
    ``shared_expert_gate``), plus the modules the MoE names reach under peft's list
    rule; the parameters are every routed-expert tensor. ``all-linear`` means every
    ``nn.Linear`` but ``lm_head``; a regex the modules it full-matches, or, on a
    family peft converts, every ``nn.Linear`` sharing a last name with one of those.
    Returns ``(modules, parameters, gates)``.
    """
    torch = pytest.importorskip("torch")
    model = _read_only(arch)
    named = list(model.named_modules())
    linear = {
        name for name, module in named
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
    }
    gates = {
        name for name, module in named
        if "router" in type(module).__name__.lower() or name.endswith(".shared_expert_gate")
    }
    if target_modules in ("auto", "all-linear"):
        meant = linear
    else:
        matched = {
            name for name, _ in named if name and re.fullmatch(target_modules, name)
        } - gates
        if arch in _PEFT_CONVERTS:
            leaves = {name.rsplit(".", 1)[-1] for name in matched}
            meant = {name for name in linear if name.rsplit(".", 1)[-1] in leaves}
        else:
            meant = matched
    names = moe_mod.get_moe_target_modules(model)
    modules = (meant | {name for name in linear if _peft_list_rule(names, name)}) - gates
    if arch in _PEFT_CONVERTS:
        modules = {name for name in modules if name.rsplit(".", 1)[-1] not in _V4_EXPERT_NAMES}
    parameters = {
        name for name, param in model.named_parameters()
        if param.ndim == 3 and ".experts." in name
    }
    return frozenset(modules), frozenset(parameters), frozenset(gates)


#: (arch, target_modules). ``auto`` on qwen4_exp is the reported case: it resolves to
#: ``"all-linear"``, which peft does not convert there. qwen3_moe and deepseek_v3 are
#: families peft converts.
_PLAIN_CASES = [
    ("qwen4_exp", "auto"),
    ("qwen3_moe", "all-linear"),
    ("deepseek_v3", "all-linear"),
    ("qwen4_exp", r".*\.linear_attn\.in_proj_(qkv|z)"),
    ("qwen3_moe", r"model\.layers\.0\.self_attn\.q_proj"),
    ("deepseek_v3", r".*\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)"),
    ("deepseek_v3", r"model\.layers\.0\.self_attn\.q_a_proj"),
]
#: A regex that names the routers itself: under moe_lora they are still left out.
_ROUTER_NAMING_CASES = [
    ("qwen3_moe", r".*\.mlp\.gate|.*\.self_attn\.q_proj"),
    ("deepseek_v3", r".*\.mlp\.gate|.*\.self_attn\.q_a_proj"),
    ("qwen4_exp", r".*\.mlp\.(gate|shared_expert_gate)|.*\.in_proj_z"),
]
_STRING_CASES = _PLAIN_CASES + _ROUTER_NAMING_CASES


class TestTheGatesAreFoundByStructure:
    @pytest.mark.parametrize(
        ("arch", "gates"),
        [
            ("qwen3_moe", {"model.layers.0.mlp.gate", "model.layers.1.mlp.gate"}),
            ("deepseek_v3", {"model.layers.1.mlp.gate"}),
            (
                "qwen4_exp",
                {
                    "model.layers.0.mlp.gate",
                    "model.layers.0.mlp.shared_expert_gate",
                    "model.layers.1.mlp.gate",
                    "model.layers.1.mlp.shared_expert_gate",
                },
            ),
            # DbrxRouter wraps one nn.Linear; its experts are stacked 2-D tensors, so
            # the count comes from config.ffn_config.moe_num_experts.
            ("dbrx", {"transformer.blocks.0.ffn.router", "transformer.blocks.1.ffn.router"}),
            # JetMoE has no experts child: each router sits beside 3-D expert tensors,
            # in the MLP and in the attention's mixture of experts.
            (
                "jetmoe",
                {
                    "model.layers.0.mlp.router",
                    "model.layers.0.self_attention.experts.router",
                    "model.layers.1.mlp.router",
                    "model.layers.1.self_attention.experts.router",
                },
            ),
        ],
    )
    def test_every_router_and_shared_expert_gate(self, arch, gates):
        found = moe_mod.find_moe_gates(_read_only(arch))

        assert set(found) == gates
        assert len(found) == len(gates)

    def test_the_per_expert_layout(self):
        torch = pytest.importorskip("torch")
        model = _per_expert_linear_moe(torch, shared_gate=True, attn_gate=True)

        assert moe_mod.find_moe_gates(model) == [_ROUTER, _SHARED_GATE]

    def test_a_router_beside_expert_tensors_but_no_dense_or_attention_gate(self):
        torch = pytest.importorskip("torch")

        assert moe_mod.find_moe_gates(_jetmoe_like(torch)) == [
            "mlp.router",
            "mixture.experts.router",
        ]

    def test_fused_experts_give_the_count_when_the_config_states_none(self):
        """The 3-D expert tensors alone say how many experts there are."""
        torch = pytest.importorskip("torch")

        class _Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = torch.nn.Parameter(torch.zeros(3, 16, 8))
                self.down_proj = torch.nn.Parameter(torch.zeros(3, 8, 8))

        class _TopKRouter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(3, 8))

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="custom_moe")
                self.mlp = torch.nn.ModuleDict(
                    {"gate": _TopKRouter(), "experts": _Experts(), "up": torch.nn.Linear(8, 4)}
                )

        assert moe_mod.find_moe_gates(_Model()) == ["mlp.gate"]

    def test_in_an_experts_block_the_output_decides_not_the_name(self):
        """A child with one output per expert is taken there, a ``gate_proj`` too.

        The documented trade-off: leaving such a module unadapted is the safer error.
        With any other output size the same ``gate_proj`` is not a gate.
        """
        torch = pytest.importorskip("torch")
        gate_proj = "layers.0.block_sparse_moe.gate_proj"

        assert gate_proj in moe_mod.find_moe_gates(
            _per_expert_linear_moe(torch, block_gate_proj=2)
        )
        assert gate_proj not in moe_mod.find_moe_gates(
            _per_expert_linear_moe(torch, block_gate_proj=16)
        )

    def test_a_single_expert_block_routes_nothing(self):
        torch = pytest.importorskip("torch")
        model = _per_expert_linear_moe(torch, experts=1, shared_gate=True)

        assert moe_mod.find_moe_gates(model) == []

    def test_a_dense_model_and_a_stand_in_have_none(self):
        assert moe_mod.find_moe_gates(_read_only("dense")) == []
        assert moe_mod.find_moe_gates(_FakeMoE()) == []


class TestTheStringMinusTheGatesAttaches:
    @pytest.mark.parametrize(("arch", "target_modules"), _STRING_CASES)
    def test_the_adapted_set_is_the_expected_one(self, arch, target_modules):
        modules, params, _ = _attach(arch, target_modules, moe_lora=True)
        expected_modules, expected_params, _ = _expected(arch, target_modules)

        assert expected_params
        assert modules == expected_modules
        assert params == expected_params

    @pytest.mark.parametrize(("arch", "target_modules"), _STRING_CASES)
    def test_no_router_gate_or_output_head_is_adapted(self, arch, target_modules):
        modules, params, _ = _attach(arch, target_modules, moe_lora=True)
        _, _, gates = _expected(arch, target_modules)

        assert gates
        assert not gates & modules
        assert not {param.rsplit(".", 1)[0] for param in params} & gates
        assert not any(name.endswith("lm_head") for name in modules)

    @pytest.mark.parametrize("arch", ["dbrx", "jetmoe"])
    def test_a_router_that_wraps_a_linear_is_left_out(self, arch):
        """DBRX and JetMoE: all-linear is kept, the routers' Linears are excluded."""
        torch = pytest.importorskip("torch")
        modules, params, targets = _attach(arch, "all-linear", moe_lora=True)
        model = _read_only(arch)
        linear = {
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name != "lm_head"
        }
        router_linears = {name for name in linear if ".router." in name}

        assert router_linears
        assert isinstance(targets, MoeLoraTargetString) and targets == "all-linear"
        assert set(targets.exclude_modules) == router_linears
        assert modules == linear - router_linears
        assert not any(".router." in param for param in params)

    @pytest.mark.parametrize(("arch", "target_modules"), _PLAIN_CASES)
    def test_without_moe_lora_the_string_reaches_peft_untouched(self, arch, target_modules):
        """The same attach as handing peft the resolved string directly, as on main."""
        from peft import LoraConfig

        from soup_cli.utils.peft_wiring import resolve_lora_target_modules

        modules, params, targets = _attach(arch, target_modules, moe_lora=False)
        model = _build(arch)
        string = resolve_lora_target_modules(model, target_modules)
        direct = _peft_attach(
            model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=string)
        )

        assert type(targets) is str and targets == string
        assert (set(modules), set(params)) == direct

    def test_an_explicit_list_naming_the_router_is_honoured(self):
        """A list is extended, never filtered: listing ``gate`` adapts the router."""
        _, params, targets = _attach_uncached("qwen3_moe", ["q_proj", "gate"])

        assert isinstance(targets, MoeLoraTargets)
        assert list(targets)[:2] == ["q_proj", "gate"]
        assert {"model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"} <= params

    def test_without_moe_lora_peft_still_adapts_the_routers(self):
        _, qwen3_params, _ = _attach("qwen3_moe", "all-linear", moe_lora=False)
        dbrx_modules, _, _ = _attach("dbrx", "all-linear", moe_lora=False)

        assert {"model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"} <= (
            qwen3_params
        )
        assert "transformer.blocks.0.ffn.router.layer" in dbrx_modules

    def test_qwen4_exp_gets_back_every_linear_the_moe_list_dropped(self):
        modules, _, targets = _attach("qwen4_exp", "auto", moe_lora=True)
        moe_modules, _, _ = _attach("qwen4_exp", None, moe_lora=True)
        leaves = {name.rsplit(".", 1)[-1] for name in modules}

        assert targets == "all-linear"
        assert len(modules) == 32 and len(moe_modules) == 10
        assert {
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "index_qk_proj",
            "input_mix_weight_down",
            "input_mix_weight_up",
            "block_inject_weight",
            "key_proj",
            "value_proj",
        } <= leaves
        assert "shared_expert_gate" not in leaves

    def test_the_regex_reaches_peft_as_one_plain_string(self):
        from soup_cli.utils.attach_preflight import trainer_lora_config

        user_regex = r".*\.linear_attn\.in_proj_(qkv|z)"
        cfg = SimpleNamespace(
            task="sft", modality="text", training=_tcfg(user_regex, moe_lora=True)
        )
        peft_config, _ = trainer_lora_config(_build("qwen4_exp"), cfg)

        assert type(peft_config.target_modules) is str
        assert peft_config.target_modules == rf"(?:{user_regex})|(?:{_NAMES_REGEX})"
        assert not peft_config.exclude_modules
        assert sorted(peft_config.target_parameters) == _FUSED


class TestThePerStringChoice:
    def test_qwen4_exp_keeps_the_string_and_excludes_the_shared_gates(self):
        """peft does not convert qwen4_exp: the string stays, ``exclude_modules`` acts."""
        from soup_cli.utils.attach_preflight import trainer_lora_config

        cfg = SimpleNamespace(task="sft", modality="text", training=_tcfg("auto"))
        peft_config, targets = trainer_lora_config(_build("qwen4_exp"), cfg)

        assert isinstance(targets, MoeLoraTargetString) and targets == "all-linear"
        assert peft_config.target_modules == "all-linear"
        assert set(peft_config.exclude_modules) == {
            "model.layers.0.mlp.shared_expert_gate",
            "model.layers.1.mlp.shared_expert_gate",
        }

    def test_exclude_modules_is_what_keeps_the_shared_gates_out(self):
        """The same config, minus ``exclude_modules``, adapts both shared gates."""
        from soup_cli.utils.attach_preflight import trainer_lora_config

        cfg = SimpleNamespace(task="sft", modality="text", training=_tcfg("auto"))
        peft_config, _ = trainer_lora_config(_build("qwen4_exp"), cfg)
        peft_config.exclude_modules = None
        modules, _ = _peft_attach(_build("qwen4_exp"), peft_config)

        assert {
            "model.layers.0.mlp.shared_expert_gate",
            "model.layers.1.mlp.shared_expert_gate",
        } <= modules

    @pytest.mark.parametrize("arch", ["qwen3_moe", "deepseek_v3"])
    def test_all_linear_on_a_family_peft_converts_becomes_a_list(self, arch):
        _, _, targets = _attach(arch, "all-linear", moe_lora=True)

        assert isinstance(targets, MoeLoraTargets)
        assert "gate" not in targets

    @pytest.mark.parametrize(
        ("arch", "routers"),
        [
            ("qwen3_moe", {"model.layers.0.mlp.gate", "model.layers.1.mlp.gate"}),
            ("deepseek_v3", {"model.layers.1.mlp.gate"}),
        ],
    )
    def test_a_regex_naming_the_routers_stays_a_string_and_excludes_them(self, arch, routers):
        """peft's resolution of a regex skips an excluded module, so no list is needed."""
        regex = dict(_ROUTER_NAMING_CASES)[arch]
        _, params, targets = _attach(arch, regex, moe_lora=True)

        assert isinstance(targets, MoeLoraTargetString)
        assert set(targets.exclude_modules) == routers
        assert not {f"{router}.weight" for router in routers} & params

    @pytest.mark.parametrize("arch", ["qwen3_moe", "deepseek_v3"])
    def test_a_regex_reaching_no_router_stays_a_string(self, arch):
        _, _, targets = _attach(arch, dict(_PLAIN_CASES[4:])[arch], moe_lora=True)

        assert isinstance(targets, MoeLoraTargetString)
        assert not targets.exclude_modules

    def test_the_list_uses_leaf_names_in_model_order(self):
        _, _, targets = _attach("qwen3_moe", "all-linear", moe_lora=True)

        # The attention leaves in model order, then the MoE names it does not repeat
        # (``get_moe_target_modules``'s fallback order on a fused-only model).
        assert list(targets) == [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]


class TestTheAnnouncementOnRealModels:
    @staticmethod
    def _line(arch, target_modules):
        from rich.console import Console

        from soup_cli.utils.peft_wiring import resolve_lora_target_modules

        model = _read_only(arch)
        buf = StringIO()
        resolve_moe_lora_targets(
            model,
            _tcfg(target_modules),
            resolve_lora_target_modules(model, target_modules),
            Console(file=buf, width=400),
        )
        return _plain(buf.getvalue())

    def test_a_list_made_of_all_linear_is_counted(self):
        assert self._line("qwen3_moe", "all-linear") == (
            "ScatterMoE LoRA: targeting 'all-linear' as a list of 7 target names: every "
            "linear layer but the output head, without the 2 MoE router/gate module(s), "
            "plus the 7 MoE module patterns and 2 fused expert parameter(s): "
            "mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )

    def test_a_kept_regex_counts_the_routers_it_excludes(self):
        assert self._line("qwen3_moe", dict(_ROUTER_NAMING_CASES)["qwen3_moe"]) == (
            "ScatterMoE LoRA: targeting the target_modules regex or the 7 MoE module "
            "patterns (one combined regex), excluding 2 MoE router/gate module(s) and 2 "
            "fused expert parameter(s): mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )

    def test_a_kept_string_counts_what_it_excludes(self):
        assert self._line("qwen4_exp", "auto") == (
            "ScatterMoE LoRA: keeping target_modules 'all-linear' (it already covers the 7 "
            "MoE module patterns), excluding 2 MoE router/gate module(s) and adding 2 fused "
            "expert parameter(s): mlp.experts.down_proj, mlp.experts.gate_up_proj"
        )


# --------------------------------------------------------------------------- #
# Layer streaming hands its resolved targets to the same helper (#2072)        #
# --------------------------------------------------------------------------- #


class TestLayerStreamingTakesTheSamePath:
    """The streamed LoRA config attaches what the resident one does, string or list.

    Driven through the shared streaming harness of ``test_stream_moe_lora_targets``:
    the setup's meta-skeleton probe is the tiny real model, so the resolver and this
    helper see a real architecture.
    """

    @staticmethod
    def _resident(build, lora):
        from soup_cli.utils.attach_preflight import trainer_lora_config

        cfg = SimpleNamespace(
            task="sft", modality="text", training=SimpleNamespace(moe_lora=True, lora=lora)
        )
        peft_config, _ = trainer_lora_config(build(), cfg)
        return _peft_attach(build(), peft_config)

    @pytest.mark.parametrize(
        ("label", "real_helpers", "kept"),
        [
            ("qwen3_5_moe", True, True),
            # The harness swaps peft's ``LoraConfig`` for a stub, which peft's own
            # conversion cannot run on: the helper gets no answer, and fails closed to
            # a list.
            ("qwen3_5_moe", False, False),
            ("deepseek_v3-q_lora_rank-int", True, False),
        ],
        ids=["qwen3_5_moe-kept-string", "qwen3_5_moe-no-peft-helpers-list", "deepseek_v3-list"],
    )
    def test_all_linear_streams_as_it_attaches_resident(
        self, tmp_path, monkeypatch, label, real_helpers, kept
    ):
        pytest.importorskip("torch")
        pytest.importorskip("peft")
        from tests import test_stream_moe_lora_targets as harness

        if real_helpers:
            # Bound to the real peft now, before the harness stubs it.
            expand = moe_mod._peft_all_linear_expander()
            convert = moe_mod._peft_config_converter()
            monkeypatch.setattr(moe_mod, "_peft_all_linear_expander", lambda: expand)
            monkeypatch.setattr(moe_mod, "_peft_config_converter", lambda: convert)
        stream_arch, build = harness._BASES[label]
        lora = harness._lora("all-linear")
        streamed, _probes = harness._streamed(tmp_path, monkeypatch, stream_arch, build(), lora)
        kwargs = dict(vars(streamed))
        monkeypatch.undo()  # the harness stubs peft; attach with the real one
        from peft import LoraConfig

        streamed_set = _peft_attach(build(), LoraConfig(**kwargs))
        resident_set = self._resident(build, lora)
        gates = moe_mod.find_moe_gates(build())
        modules, params = streamed_set

        assert gates
        assert streamed_set == resident_set
        assert not [name for name in modules if moe_mod._in_a_gate(name, gates)]
        assert not [param for param in params if moe_mod._in_a_gate(param, gates)]
        assert any(".mlp.experts." in param for param in params)
        if kept:
            assert kwargs["target_modules"] == "all-linear"
            assert kwargs["exclude_modules"]
            assert all(moe_mod._in_a_gate(name, gates) for name in kwargs["exclude_modules"])
        else:
            assert isinstance(kwargs["target_modules"], list)
            assert "gate" not in kwargs["target_modules"]
