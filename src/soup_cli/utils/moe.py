"""MoE (Mixture of Experts) model detection and LoRA target module helpers.

Supports: Qwen3 MoE (30B-A3B), Mixtral (8x7B, 8x22B), DeepSeek V3, DBRX,
OLMoE, JetMoE, and other models using MoE / sparse expert architectures.
"""

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
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if not isinstance(model_type, str):
        return False

    try:
        from peft import LoraConfig
        from peft.utils.transformers_weight_conversion import (
            convert_peft_config_for_transformers,
        )
    except ImportError:
        return False

    # A throwaway config, so the conversion's in-place rewrite touches nothing
    # the caller owns.
    probe = LoraConfig(r=8, target_modules=list(target_modules or []))
    try:
        convert_peft_config_for_transformers(probe, model, None)
    except Exception:  # noqa: BLE001 - a probe must never break the run
        return False
    return bool(getattr(probe, "target_parameters", None))


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

    Raises:
        ValueError: fused expert parameters will be targeted AND ``lora.dropout``
            is non-zero. peft adapts those through ``lora.ParamWrapper``, which
            refuses any dropout; letting the attach proceed gets the user peft's
            message with no mention of the flag that caused it.
    """
    if not getattr(tcfg, "moe_lora", False):
        return target_modules
    if not detect_moe_model(model):
        return target_modules
    moe_targets = get_moe_target_modules(model)
    if not moe_targets:
        return target_modules
    fused = find_fused_expert_parameters(model)
    modules = list(target_modules) if isinstance(target_modules, list) else []
    modules.extend(name for name in moe_targets if name not in modules)
    _refuse_dropout_on_fused_experts(model, tcfg, modules, fused)
    if console is not None:
        line = f"[green]ScatterMoE LoRA:[/] targeting {len(modules)} module patterns"
        if fused:
            line += f" and {len(fused)} fused expert parameter(s): {', '.join(fused)}"
        console.print(line)
    return MoeLoraTargets(modules, fused)


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
