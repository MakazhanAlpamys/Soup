"""MoE (Mixture of Experts) model detection and LoRA target module helpers.

Supports: Qwen3 MoE (30B-A3B), Mixtral (8x7B, 8x22B), DeepSeek V3, DBRX,
OLMoE, JetMoE, and other models using MoE / sparse expert architectures.
"""

import re
from typing import Optional

# Known MoE architecture config keys that indicate expert layers
MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "n_routed_experts",
    "moe_num_experts",
)

# Common expert FFN module name patterns across MoE architectures
MOE_EXPERT_PATTERNS = [
    "experts",          # Mixtral, Qwen3, DeepSeek
    "gate_proj",        # Expert gate projection
    "up_proj",          # Expert up projection
    "down_proj",        # Expert down projection
    "w1",              # DeepSeek V3 expert naming
    "w2",
    "w3",
]

# Standard attention + MLP target modules (non-expert layers)
STANDARD_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

_MOE_MODEL_TYPES = {
    "mixtral",
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen4_exp_text",
    "qwen2_moe",
    "dbrx",
    "deepseek_v2",
    "deepseek_v3",
    "olmoe",
    "jetmoe",
    "arctic",
    "grok",
}


def _candidate_configs(model):
    if not hasattr(model, "config"):
        return ()
    config = model.config
    text_config = None
    if hasattr(config, "__dict__"):
        text_config = config.__dict__.get("text_config")
    return tuple(cfg for cfg in (text_config, config) if cfg is not None)


def detect_moe_model(model) -> bool:
    """Detect whether a model uses Mixture of Experts architecture.

    Checks model config for known MoE indicators.
    """
    for config in _candidate_configs(model):
        # Check for known MoE config keys
        for key in MOE_CONFIG_KEYS:
            value = getattr(config, key, None)
            if value is not None and isinstance(value, (int, float)) and value > 1:
                return True

        # Check model_type for known MoE architectures
        model_type = getattr(config, "model_type", "")
        if isinstance(model_type, str) and model_type.lower() in _MOE_MODEL_TYPES:
            return True

    return False


def get_moe_target_modules(model) -> Optional[list[str]]:
    """Get LoRA target modules for MoE models (ScatterMoE LoRA).

    Returns a list of module name patterns that includes both attention
    layers and expert FFN layers for comprehensive LoRA coverage.
    Returns None if the model is not an MoE model.
    """
    if not detect_moe_model(model):
        return None

    # Scan model for expert layer names
    expert_modules = set()
    for name, _module in model.named_modules():
        name_lower = name.lower()
        # Look for expert-specific patterns
        if "expert" in name_lower or "moe" in name_lower:
            parts = name.split(".")
            for part in parts:
                if part in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"):
                    expert_modules.add(part)

    # Combine attention targets with discovered expert targets
    targets = list(STANDARD_TARGET_MODULES)
    if expert_modules:
        targets.extend(sorted(expert_modules))
    else:
        # Fallback: add common expert FFN patterns
        targets.extend(["gate_proj", "up_proj", "down_proj"])

    return targets


def find_fused_expert_parameters(model) -> list[str]:
    """Fused routed-expert tensors, as the suffixes peft's ``target_parameters`` match.

    On transformers 5 an MoE layer holds all its experts as 3-D parameters on one
    module (``mlp.experts.gate_up_proj`` of shape ``(experts, in, out)``), not as
    one ``nn.Linear`` per expert, so module-name discovery cannot reach them
    (#1421). This looks at the parameters: a 3-D tensor whose module path has an
    ``experts`` component. The suffix starts at the component before ``experts``
    (``mlp.experts.gate_up_proj``, ``block_sparse_moe.experts.down_proj``), which
    is what peft matches with ``key.endswith("." + target)``. A shared expert's
    ``gate_proj.weight`` is 2-D and is not one of these.

    A stand-in with no parameters (a namespace or a mock) yields ``[]``.
    """
    try:
        named = list(model.named_parameters())
    except Exception:  # noqa: BLE001 - a stand-in for a model has no parameters
        return []
    found = set()
    for name, param in named:
        if getattr(param, "ndim", 0) != 3 or not isinstance(name, str):
            continue
        parts = name.split(".")
        expert_indices = [i for i, part in enumerate(parts[:-1]) if "experts" in part.lower()]
        if not expert_indices:
            continue
        start = max(expert_indices[-1] - 1, 0)
        found.add(".".join(parts[start:]))
    return sorted(found)


class MoeLoraTargets(list):
    """The module targets ``moe_lora`` resolved, plus the fused expert parameters.

    A list, so every trainer's ``target_modules`` plumbing is unchanged; the
    ``target_parameters`` ride along and :func:`peft_wiring.build_lora_config`
    merges them into the ``LoraConfig`` -- the one call every trainer makes before
    ``get_peft_model``, the same route :class:`peft_wiring.UnmappedTargets` takes.
    """

    __slots__ = ("target_parameters",)

    def __init__(self, modules, target_parameters=()):
        super().__init__(modules)
        self.target_parameters = list(target_parameters)


class MoeLoraTargetString(str):
    """A string ``target_modules`` ``moe_lora`` kept, plus what rides along with it.

    The string twin of :class:`MoeLoraTargets`: peft's ``"all-linear"`` as written,
    or a user regex joined with the MoE module names into one regex. A ``str``, so
    every trainer and peft read it as the string target it is. The fused expert
    ``target_parameters`` and the MoE router / gate modules the string would
    otherwise reach (``exclude_modules``) ride along to
    :func:`peft_wiring.build_lora_config`, which hands peft a plain ``str``.
    """

    __slots__ = ("target_parameters", "exclude_modules")

    def __new__(cls, value, target_parameters=(), exclude_modules=()):
        instance = super().__new__(cls, value)
        instance.target_parameters = list(target_parameters)
        instance.exclude_modules = list(exclude_modules)
        return instance


#: peft's shorthand for every ``nn.Linear`` / ``Conv1D`` but the output head
#: (``peft.utils.constants.INCLUDE_LINEAR_LAYERS_SHORTHAND``), which peft compares
#: lower-cased, so ``ALL-LINEAR`` is the same target.
ALL_LINEAR = "all-linear"

#: A run of global inline flags at the start of a regex: ``(?i)``, ``(?x)(?s)``.
_LEADING_GLOBAL_FLAGS = re.compile(r"(?:\(\?[aiLmsux]+\))+")


def _names_as_full_match_regex(names) -> str:
    """peft's LIST rule as one full-match regex: the key is a name or ends ``.<name>``."""
    return r"(?:.*\.)?(?:" + "|".join(re.escape(name) for name in names) + ")"


def _join_regex_with_names(user_regex: str, names) -> str:
    """``(?:<user regex>)|(?:<names>)``: what either half matches, as peft full-matches.

    peft matches a string ``target_modules`` with ``re.fullmatch`` and a list entry
    with ``key == name or key.endswith("." + name)``; no ``target_modules`` value
    mixes the two, so the names are spelled as the regex equivalent of that rule.
    Python accepts global inline flags (``(?i)``, ``(?x)``) only at the start of a
    pattern, so a leading run of them is hoisted out of the user's group, and the
    names half is shielded from them with ``(?-imsx:...)``. Under ``(?x)`` a newline
    closes the user's half first, so a trailing ``# comment`` cannot swallow the
    closing parenthesis. ``user_regex`` has passed :func:`_check_user_regex`.
    """
    flags_match = _LEADING_GLOBAL_FLAGS.match(user_regex)
    flags = flags_match.group(0) if flags_match else ""
    body = user_regex[len(flags):]
    if "x" in flags:
        body += "\n"
    names_regex = _names_as_full_match_regex(names)
    names_group = f"(?-imsx:{names_regex})" if flags else f"(?:{names_regex})"
    return f"{flags}(?:{body})|{names_group}"


def _check_user_regex(user_regex: str) -> None:
    """Refuse a regex that does not compile, by its field name.

    Rather than let it surface from peft, joined with names the user never wrote.
    """
    try:
        re.compile(user_regex)
    except re.error as exc:
        raise ValueError(
            f"training.lora.target_modules is not a valid regular expression ({exc}). "
            "peft full-matches a string target_modules against every module name, and "
            "training.moe_lora adds the MoE module names to it. Fix the regex, or give "
            "target_modules as a list of module names."
        ) from exc


#: Printed when a regex target full-matches no module path. The regex itself is not
#: printed: a character class such as ``[qkv]`` is a Rich tag.
_NO_MODULE_MATCHED = (
    "[yellow]training.lora.target_modules is a regex that matches no module of this "
    "model:[/] peft full-matches it against each module's full path, so only "
    "moe_lora's own targets attach. To name modules by their last name, give "
    "target_modules as a list."
)


def _matches_a_module(model, user_regex: str) -> bool:
    try:
        names = [name for name, _ in model.named_modules() if name]
    except Exception:  # noqa: BLE001 - a stand-in for a model has no modules
        return True
    return any(re.fullmatch(user_regex, name) for name in names)


def _string_target_kind(target_modules) -> Optional[str]:
    """``"all-linear"``, ``"regex"``, or ``None`` for a target that is not a string.

    ``auto`` is Soup's unresolved placeholder ("decide for me"), not a peft target,
    so it is treated like no targets at all, as before.
    """
    if not isinstance(target_modules, str) or target_modules == "auto":
        return None
    if target_modules.lower() == ALL_LINEAR:
        return "all-linear"
    return "regex"


def peft_routes_lora_to_fused_params(model, target_modules) -> bool:
    """Will peft turn THESE targets into ``target_parameters`` on THIS model?

    Asks peft rather than guessing, because a model-wide "are there fused 3-D
    expert parameters" scan gets the answer wrong. Measured on tiny stand-ins,
    transformers 5.16.1 / peft 0.20.0, with the targets
    ``get_moe_target_modules`` produces:

    ================  ===================  =========================  ==============
    architecture      fused 3-D params     attach at dropout 0.05     expert adapters
    ================  ===================  =========================  ==============
    qwen3_moe         yes                  ValueError, ParamWrapper   2
    deepseek_v3       yes                  ValueError, ParamWrapper   2
    glm4_moe          yes                  ValueError, ParamWrapper   2
    mixtral           yes                  attaches fine              0
    minimax           yes                  attaches fine              0
    ================  ===================  =========================  ==============

    All five have ``mlp.experts.gate_up_proj`` and ``mlp.experts.down_proj`` as
    3-D parameters, and identical module structure, so the model alone cannot
    tell these rows apart. What differs is inside peft:
    ``utils/transformers_weight_conversion.convert_peft_config_for_transformers``
    rewrites ``target_modules`` into ``target_parameters`` only when the model
    type has a v4->v5 checkpoint conversion mapping. That rewrite is what
    produces a ``lora.ParamWrapper``, and the wrapper is what refuses dropout.
    Mixtral and MiniMax have no mapping, so their experts are never targeted --
    they attach happily, and adapt no experts at all (see #1070).

    So the probe runs peft's own conversion over a throwaway config and looks
    at what came out. If that private path is not importable -- a different peft
    -- this returns False: Soup then stays quiet and peft raises its own error,
    which is worse UX but strictly better than refusing a config that works.
    """
    # The conversion keys off ``model.config.model_type``. A stub or a MagicMock
    # answers every attribute, so peft's lookup succeeds against nothing real and
    # the probe comes back True for a model that has no experts at all; a real
    # architecture always has a plain string here.
    return bool(_peft_converted_parameters(model, target_modules))


def _peft_config_converter():
    """peft's private v4->v5 conversion, as ``(model, targets, excluded) -> parameters``.

    ``None`` where peft or that private path is not importable. The probe config is a
    throwaway, so the conversion's in-place rewrite touches nothing the caller owns.
    """
    try:
        from peft import LoraConfig
        from peft.utils.transformers_weight_conversion import (
            convert_peft_config_for_transformers,
        )
    except ImportError:
        return None

    def convert(model, target_modules, exclude_modules):
        probe = LoraConfig(
            r=8, target_modules=target_modules, exclude_modules=list(exclude_modules) or None
        )
        convert_peft_config_for_transformers(probe, model, None)
        return set(getattr(probe, "target_parameters", None) or ())

    return convert


def _peft_all_linear_expander():
    """peft's private ``all-linear`` expansion, as ``model -> module names``.

    ``None`` where peft or that private path is not importable.
    """
    try:
        from peft import LoraConfig
        from peft.tuners.tuners_utils import _maybe_include_all_linear_layers
    except ImportError:
        return None

    def expand(model):
        probe = LoraConfig(r=8, target_modules=ALL_LINEAR)
        return set(_maybe_include_all_linear_layers(probe, model).target_modules)

    return expand


def _peft_converted_parameters(model, target_modules, exclude_modules=()) -> Optional[set]:
    """The ``target_parameters`` peft's v4->v5 conversion makes of these targets.

    Empty when the model has no plain-string ``model_type`` (a stub answers every
    attribute, so peft's lookup would succeed against nothing real). ``None`` when
    the answer is unknown: peft, or its private conversion, is not importable, or
    the conversion raises.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if not isinstance(model_type, str):
        return set()
    convert = _peft_config_converter()
    if convert is None:
        return None
    # A string stays a string: peft's conversion resolves ``"all-linear"`` or a
    # regex against the model, where ``list()`` of it would be one target per
    # character. The exclusions go in too: peft's resolution of a regex skips an
    # excluded module, its ``all-linear`` does not.
    if not isinstance(target_modules, str):
        target_modules = list(target_modules or [])
    try:
        return convert(model, target_modules, exclude_modules)
    except Exception:  # noqa: BLE001 - a probe must never break the run
        return None


def _named_children(module) -> dict:
    try:
        return dict(module.named_children())
    except Exception:  # noqa: BLE001 - a stand-in for a module has no children
        return {}


def _parameters(module, *, recurse: bool) -> dict:
    try:
        return dict(module.named_parameters(recurse=recurse))
    except Exception:  # noqa: BLE001 - a stand-in for a module has no parameters
        return {}


#: Config keys that state a model's TOTAL routed-expert count (never its top-k).
_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")

#: Fewer experts than this is no routing at all.
_MIN_EXPERTS = 2


def _config_expert_counts(model) -> set:
    """The expert counts the config states, its sub-configs one level down included.

    DBRX keeps its count in ``config.ffn_config.moe_num_experts`` and its experts as
    2-D tensors (experts x hidden stacked into one dimension), so no shape gives it.
    """
    counts = set()
    for config in _candidate_configs(model):
        holders = [config]
        holders.extend(
            value for value in getattr(config, "__dict__", {}).values()
            if hasattr(value, "__dict__")
        )
        for holder in holders:
            for key in _EXPERT_COUNT_KEYS:
                value = getattr(holder, key, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    counts.add(value)
    return {count for count in counts if count >= _MIN_EXPERTS}


def _experts_counts(experts) -> set:
    """Expert counts an ``experts`` module shows: 3-D leading dims, or its per-expert list."""
    counts = {
        int(param.shape[0])
        for param in _parameters(experts, recurse=True).values()
        if getattr(param, "ndim", 0) == 3
    }
    children = list(_named_children(experts).values())
    if len({type(child) for child in children}) == 1:
        counts.add(len(children))
    return {count for count in counts if count >= _MIN_EXPERTS}


def _output_dim(module) -> Optional[int]:
    """``out_features``, the rows of the module's own 2-D ``weight``, or its sole Linear's.

    DBRX's ``DbrxRouter`` and JetMoE's ``JetMoeTopKGating`` wrap one ``nn.Linear``.
    """
    out_features = getattr(module, "out_features", None)
    if isinstance(out_features, int):
        return out_features
    weight = _parameters(module, recurse=False).get("weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[0])
    children = list(_named_children(module).values())
    if len(children) == 1:
        out_features = getattr(children[0], "out_features", None)
        if isinstance(out_features, int):
            return out_features
    return None


#: The names a router goes by when its block has no ``experts`` child (JetMoE).
_ROUTER_NAMES = ("router", "gate")


def find_moe_gates(model) -> list[str]:
    """Module paths of every MoE router and shared-expert gate, found by structure.

    A router's output is one logit per routed expert; a shared-expert gate's is the
    single logit that scales the shared expert (``shared_expert_gate`` on Qwen2-MoE,
    Qwen3-Next and qwen4_exp). Both are routing logits, so under ``moe_lora`` neither
    gets an adapter from a string target. Two layouts:

    * a block with an ``experts`` child: the router is another child whose output
      dimension is the expert count, the shared-expert gate one whose output
      dimension is 1. The count is read from the experts (the leading dimension of
      their 3-D tensors, or the number of per-expert modules) or, where the experts
      are stacked 2-D tensors (DBRX), from the config.
    * a block without one (JetMoE's ``mlp`` and its attention's mixture): a child
      named ``router`` or ``gate`` whose output dimension is the leading dimension of
      a sibling's own 3-D expert tensor.

    A module's output dimension is its ``out_features``, the rows of its own 2-D
    ``weight`` (the transformers-5 ``TopKRouter`` holds a bare parameter), or the
    ``out_features`` of its sole child (a router that wraps one ``nn.Linear``).

    Only the second layout looks at names, so there a ``gate_proj`` is never a
    candidate, nor is an attention ``gate`` with no expert tensor beside it. The
    first is decided by output dimension alone: a direct child of an ``experts``
    block with one output, or one per expert, is taken whatever its name, a
    ``gate_proj`` included (no known family has one there). Leaving such a module
    unadapted is the safer error; a name guard would adapt a router that happened
    to carry the name. Model order; a stand-in with no modules yields ``[]``.
    """
    try:
        named = list(model.named_modules())
    except Exception:  # noqa: BLE001 - a stand-in for a model has no modules
        return []
    config_counts = _config_expert_counts(model)
    gates = []
    for path, module in named:
        children = _named_children(module)
        prefix = f"{path}." if path else ""
        experts = children.get("experts")
        if experts is not None:
            counts = _experts_counts(experts) or config_counts
            if not counts:
                continue
            for name, child in children.items():
                if name != "experts" and (
                    _output_dim(child) in counts or _output_dim(child) == 1
                ):
                    gates.append(prefix + name)
            continue
        expert_dims = {
            int(param.shape[0])
            for name, child in children.items()
            if name not in _ROUTER_NAMES
            for param in _parameters(child, recurse=False).values()
            if getattr(param, "ndim", 0) == 3 and int(param.shape[0]) >= _MIN_EXPERTS
        }
        for name in _ROUTER_NAMES:
            if name in children and _output_dim(children[name]) in expert_dims:
                gates.append(prefix + name)
    return gates


def _in_a_gate(name, gates) -> bool:
    """The module is a gate, or inside one (a router's wrapped ``nn.Linear``)."""
    return any(name == gate or name.startswith(f"{gate}.") for gate in gates)


def _gate_parameter_keys(model, gates) -> list[str]:
    modules = dict(model.named_modules())
    return [
        f"{gate}.{name}"
        for gate in gates
        for name in _parameters(modules.get(gate), recurse=True)
    ]


def _lands_on(produced, parameter_keys) -> bool:
    """Would these ``target_parameters`` adapt any of these parameters?

    The same match peft's ``_inject_parameters`` makes: the key, or a ``.<name>``
    suffix of it.
    """
    return any(
        key == name or key.endswith(f".{name}") for key in parameter_keys for name in produced
    )


def _all_linear_modules(model) -> Optional[set]:
    """peft's own ``all-linear``: every ``nn.Linear`` / ``Conv1D`` but the output head.

    ``None`` when peft's private expansion is not importable, or raises (a private
    signature is not a promise): the caller then fails closed, as for the conversion.
    """
    expand = _peft_all_linear_expander()
    if expand is None:
        return None
    try:
        return expand(model)
    except Exception:  # noqa: BLE001 - a changed private helper must not break setup
        return None


def _linear_modules_by_type(model) -> set:
    """peft's ``all-linear`` rule without peft: Linear / Conv1D modules but the output head."""
    head = None
    get_head = getattr(model, "get_output_embeddings", None)
    if callable(get_head):
        try:
            head = get_head()
        except Exception:  # noqa: BLE001 - a model without an output head
            head = None
    return {
        name
        for name, module in model.named_modules()
        if name
        and module is not head
        and (
            isinstance(getattr(module, "out_features", None), int)
            or type(module).__name__ == "Conv1D"
        )
    }


def _compact_names(module_names, meant) -> list[str]:
    """``meant`` as list entries, a leaf name where it reaches nothing else.

    peft matches a list entry by ``key == entry or key.endswith("." + entry)``. A
    leaf name is used when every module it reaches is in ``meant`` (so it never
    names a router, the output head or anything outside the string's meaning);
    otherwise that leaf's modules are listed by full path. Model order, no
    duplicates.
    """
    meant_set = set(meant)
    by_leaf: dict = {}
    for name in meant:
        by_leaf.setdefault(name.rsplit(".", 1)[-1], []).append(name)
    entries = []
    for leaf, paths in by_leaf.items():
        reach = {name for name in module_names if name == leaf or name.endswith(f".{leaf}")}
        if reach <= meant_set:
            entries.append(leaf)
        else:
            entries.extend(paths)
    return entries


def _materialise(model, kind, target_modules, gates, moe_targets) -> list[str]:
    """The string as an explicit list: its modules but the gates, then the MoE names.

    ``all-linear`` is peft's own expansion (or, without peft's private helper, the
    same rule applied by type); a regex is the modules it full-matches. peft's
    conversion rewrites a list entry by its last name, so an entry that would still
    land on a gate's parameter that way (a non-router module whose leaf is the
    router's v4 name, ``gate``) is dropped rather than let it adapt every router.
    The list is probed once; entries are probed one by one only when it lands. With
    no answer from peft, an entry whose leaf name is a gate's is dropped.
    """
    module_names = [name for name, _ in model.named_modules() if name]
    if kind == "all-linear":
        linear = _all_linear_modules(model)
        if linear is None:
            linear = _linear_modules_by_type(model)
        meant = [name for name in module_names if name in linear]
    else:
        meant = [name for name in module_names if re.fullmatch(target_modules, name)]
    meant = [name for name in meant if not _in_a_gate(name, gates)]
    entries = _compact_names(module_names, meant)
    entries.extend(name for name in moe_targets if name not in entries)
    gate_keys = _gate_parameter_keys(model, gates)
    produced = _peft_converted_parameters(model, entries)
    if produced is None:
        gate_leaves = {gate.rsplit(".", 1)[-1] for gate in gates}
        return [entry for entry in entries if entry.rsplit(".", 1)[-1] not in gate_leaves]
    if not _lands_on(produced, gate_keys):
        return entries
    return [
        entry
        for entry in entries
        if not _lands_on(_peft_converted_parameters(model, [entry]) or (), gate_keys)
    ]


def _refuse_dropout_on_fused_experts(model, tcfg, target_modules, fused_parameters=()) -> None:
    """Stop before peft does, naming the flag that caused it (#798).

    Only when fused parameters will really be targeted: the ones Soup now names
    itself (#1421), or the ones peft's own name conversion rewrites the module
    targets into. Until #1421 Mixtral and MiniMax were not refused, because
    nothing reached their experts; now that ``moe_lora`` targets them, peft's
    ``ParamWrapper`` refuses dropout there too.
    """
    dropout = getattr(getattr(tcfg, "lora", None), "dropout", 0) or 0
    if dropout == 0:
        return
    if not fused_parameters and not peft_routes_lora_to_fused_params(model, target_modules):
        return
    raise ValueError(
        f"training.moe_lora=true needs training.lora.dropout: 0.0 on this model "
        f"(got {dropout}). peft adapts its fused expert parameters through "
        f"lora.ParamWrapper, and that refuses dropout: 'lora.ParamWrapper does "
        f"not work with lora_dropout != 0.' Set lora.dropout: 0.0 to train the "
        f"experts, or remove moe_lora to keep dropout on the attention-only "
        f"adapters."
    )


def _refuse_vera_on_fused_experts(tcfg, fused_parameters) -> None:
    """``moe_lora`` + ``lora.use_vera`` can reach no fused expert (#1197).

    VeRA (peft ``VeraConfig``) wraps ``nn.Linear`` modules only and has no
    ``target_parameters`` path, so on a model whose routed experts are fused 3-D
    tensors the flag would adapt attention (and a shared expert) exactly as VeRA
    without it, while announcing expert targets. Refused on every such family,
    Mixtral and MiniMax included, as ruled on the issue; a checkpoint with one
    ``nn.Linear`` per expert is not refused, VeRA wraps those.
    """
    if not fused_parameters or not getattr(getattr(tcfg, "lora", None), "use_vera", False):
        return
    raise ValueError(
        "training.moe_lora=true does nothing with lora.use_vera: true on this model: its "
        f"routed experts are fused 3-D parameters ({', '.join(fused_parameters)}), and VeRA "
        "wraps nn.Linear modules only, so the experts would get no adapter and only the "
        "attention projections would train. Set lora.use_vera: false to adapt the experts "
        "with LoRA, or remove moe_lora to keep VeRA on the attention projections."
    )


def resolve_moe_lora_targets(model, tcfg, target_modules, console=None):
    """Return MoE-aware LoRA targets when ``training.moe_lora`` asks for them.

    THE one place the flag turns into targets (#798). ``sft`` and ``pretrain``
    grew their own copy of this block; the five preference/RL trainers had
    none, so ``moe_lora: true`` did nothing there -- and on a fused-expert MoE
    (Qwen3-MoE on transformers 5.x) ``target_modules: auto`` resolves to
    ``None``, which peft refuses outright, so the eight DPO and seven GRPO
    recipes that set the flag could not attach LoRA at all.

    Returns ``target_modules`` unchanged when the flag is off, the model is not
    MoE, or no expert modules are found, so a non-MoE base is untouched.

    Otherwise returns a :class:`MoeLoraTargets`: the module patterns already
    resolved (an ``auto`` list is kept, so Qwen3.5's ``in_proj_qkv`` /
    ``out_proj`` survive) plus the attention and expert-FFN names, and, on a
    transformers-5 model whose routed experts are fused 3-D tensors, those
    tensors as ``target_parameters`` for peft -- the only way to adapt them, and
    the one that does not depend on peft's per-architecture name conversion
    (#1421: Mixtral, MiniMax, Qwen2-MoE and Qwen3.5-MoE got attention-only LoRA).

    A STRING target is not a list to extend, and it used to be replaced by the
    MoE list, which lost what peft makes of it. It is kept, as a
    :class:`MoeLoraTargetString` carrying the same ``target_parameters``:

    * ``"all-linear"`` (what ``auto`` resolves to on qwen4_exp) as written: peft
      expands it to every ``nn.Linear`` / ``Conv1D`` but the output head, which
      already includes every module the MoE names reach. Replacing it dropped
      qwen4_exp's Gated DeltaNet, indexer, hyper-connection and PLE projections:
      10 linears attached on a tiny model, where 32 do now.
    * any other string, a regex peft full-matches against each module name, as
      one regex of the union: ``(?:<user regex>)|(?:<the MoE names>)``.

    Either way no MoE router or shared-expert gate (:func:`find_moe_gates`) gets an
    adapter from the string, even one the user's regex names: they go to peft's
    ``exclude_modules``, or, where peft's own conversion would still adapt a router
    weight with them excluded (``all-linear`` on the families peft converts) and
    with VeRA, the string becomes an explicit list of what it means without them,
    plus the MoE names (:func:`_string_targets`). To adapt a router, name it in an
    explicit ``target_modules`` list. Without ``moe_lora`` a string reaches peft
    untouched, router included where peft includes it. A regex that matches no
    module is announced, since only the MoE targets would attach.

    Raises:
        ValueError: fused expert parameters will be targeted AND ``lora.use_vera``
            is set (VeRA cannot reach them, #1197), or AND ``lora.dropout`` is
            non-zero (peft adapts them through ``lora.ParamWrapper``, which refuses
            any dropout). Either way the user gets the flag named, not peft's
            message or a quiet attention-only adapter. Also when a string target
            is a regex that does not compile.
    """
    if not getattr(tcfg, "moe_lora", False):
        return target_modules
    if not detect_moe_model(model):
        return target_modules
    moe_targets = get_moe_target_modules(model)
    if not moe_targets:
        return target_modules
    fused = find_fused_expert_parameters(model)
    kind = _string_target_kind(target_modules)
    gates, excluded, listed = [], [], False
    if kind is None:
        modules = list(target_modules) if isinstance(target_modules, list) else []
        modules.extend(name for name in moe_targets if name not in modules)
    else:
        if kind == "regex":
            _check_user_regex(target_modules)
            if console is not None and not _matches_a_module(model, target_modules):
                console.print(_NO_MODULE_MATCHED)
        modules, gates, excluded, listed = _string_targets(
            model, tcfg, kind, target_modules, moe_targets
        )
    _refuse_vera_on_fused_experts(tcfg, fused)
    _refuse_dropout_on_fused_experts(model, tcfg, modules, fused)
    if console is not None:
        console.print(
            _announcement(kind, listed, modules, moe_targets, fused, gates, excluded)
        )
    if kind is not None and not listed:
        return MoeLoraTargetString(modules, fused, excluded)
    return MoeLoraTargets(modules, fused)


def _string_targets(model, tcfg, kind, target_modules, moe_targets):
    """A string target under ``moe_lora``: its meaning, without any MoE gate.

    Returns ``(modules, gates, excluded, listed)``. ``modules`` is the string kept
    (``all-linear`` as written, or the regex joined with the MoE names) with the gate
    modules it would reach in ``excluded``, for peft's ``exclude_modules``; or, when
    that cannot keep the gates out, an explicit list (``listed``).

    Decided per string, by asking peft's own conversion of it with those exclusions
    applied. On the families peft converts from the transformers-4 layout
    (qwen3_moe, deepseek_v3, ...) peft adds the router's v4 name to ``all-linear``
    whatever is excluded, and adapts the router weight as a ``target_parameters``
    entry, which ``exclude_modules`` does not reach; a regex it resolves module by
    module, skipping what is excluded, so a regex stays a string unless it reaches a
    non-router whose leaf name is the router's (``gate``). With VeRA, whose
    ``VeraConfig`` has no ``exclude_modules``, any exclusion means a list. So does
    an unanswered question (peft's private helpers missing): fail closed.
    """
    if kind == "all-linear":
        kept = target_modules
    else:
        kept = _join_regex_with_names(target_modules, moe_targets)
    gates = find_moe_gates(model)
    if not gates:
        return kept, gates, [], False
    if kind == "all-linear":
        reached = _all_linear_modules(model)
    else:
        reached = {name for name, _ in model.named_modules() if name and re.fullmatch(kept, name)}
    if reached is None:
        return _materialise(model, kind, target_modules, gates, moe_targets), gates, [], True
    excluded = [
        name for name, _ in model.named_modules() if name in reached and _in_a_gate(name, gates)
    ]
    if getattr(getattr(tcfg, "lora", None), "use_vera", False):
        if excluded:
            return _materialise(model, kind, target_modules, gates, moe_targets), gates, [], True
        return kept, gates, [], False
    produced = _peft_converted_parameters(model, kept, excluded)
    if produced is None or _lands_on(produced, _gate_parameter_keys(model, gates)):
        return _materialise(model, kind, target_modules, gates, moe_targets), gates, [], True
    return kept, gates, excluded, False


def _announcement(kind, listed, modules, moe_targets, fused, gates, excluded) -> str:
    """The setup line: what ``moe_lora`` did to the targets, never the user's regex.

    The regex is not printed: a character class such as ``[qkv]`` is a Rich tag.
    A string turned into a list is counted, not listed.
    """
    patterns = f"the {len(moe_targets)} MoE module patterns"
    without = f"{len(gates)} MoE router/gate module(s)"
    joiner = " and"
    if listed and kind == "all-linear":
        line = (
            f"[green]ScatterMoE LoRA:[/] targeting 'all-linear' as a list of "
            f"{len(modules)} target names: every linear layer but the output head, "
            f"without the {without}, plus {patterns}"
        )
    elif listed:
        line = (
            f"[green]ScatterMoE LoRA:[/] targeting the target_modules regex as a list of "
            f"{len(modules)} target names: the modules it matches, without the {without}, "
            f"plus {patterns}"
        )
    elif kind == "all-linear":
        line = (
            f"[green]ScatterMoE LoRA:[/] keeping target_modules 'all-linear' (it already "
            f"covers {patterns})"
        )
        joiner = " and adding"
    elif kind == "regex":
        line = (
            f"[green]ScatterMoE LoRA:[/] targeting the target_modules regex or {patterns} "
            f"(one combined regex)"
        )
    else:
        line = f"[green]ScatterMoE LoRA:[/] targeting {len(modules)} module patterns"
    if excluded:
        line += f", excluding {len(excluded)} MoE router/gate module(s)"
    if fused:
        line += f"{joiner} {len(fused)} fused expert parameter(s): {', '.join(fused)}"
    return line


def get_moe_info(model) -> dict:
    """Extract MoE architecture details from a model config.

    Returns a dict with num_experts, num_active_experts, and model_type.
    Returns empty dict if not an MoE model.
    """
    if not hasattr(model, "config"):
        return {}

    info = {}
    for config in _candidate_configs(model):
        # Number of total experts
        for key in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
            value = getattr(config, key, None)
            if value is not None:
                info["num_experts"] = value
                break

        # Number of active experts per token
        for key in ("num_experts_per_tok", "num_experts_per_token", "num_selected_experts"):
            value = getattr(config, key, None)
            if value is not None:
                info["num_active_experts"] = value
                break

        if info:
            info["model_type"] = getattr(config, "model_type", "unknown")
            break

    return info
