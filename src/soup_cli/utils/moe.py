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
