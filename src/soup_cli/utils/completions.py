"""Shell completion script generators + dynamic value completers.

`soup completions <shell>` emits a sourceable bash / zsh / fish script.
The dynamic completers (``complete_recipe_name`` /
``complete_target_modules``) are exposed for use as
``shell_complete=...`` callbacks on Typer options.

Live config introspection (probe the operator's actual ``base`` model
for its layer names) lands in v0.64.1; v0.64.0 ships canonical Llama-
shape defaults that cover ~80% of common bases.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import List, Mapping, Optional, Tuple

SUPPORTED_SHELLS = frozenset({"bash", "zsh", "fish"})
_MAX_SHELL_LEN = 32

# Canonical attention/mlp module names that cover Llama / Qwen / Mistral
# / Gemma / Phi families. Returned when no ``base`` is supplied or when
# per-base introspection is unavailable / fails.
_DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    "embed_tokens",
)

# Llama-shaped attention + gated-MLP projections shared across the
# Llama / Mistral / Qwen / Gemma / Granite / Cohere families.
_LLAMA_SHAPE: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Every nn.Linear suffix exposed by the Qwen4-Exp text decoder. This includes
# QSA, Gated DeltaNet, shared-expert, PLE, and gated-residual projections. The
# routed experts use raw 3-D parameters and therefore do not belong in a
# target-*module* completer.
_QWEN4_EXP_TEXT_SHAPE: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "index_qk_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert_gate",
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
    "key_proj",
    "value_proj",
)

# Per-``model_type`` LoRA target-module names (v0.71.1 #210). Keyed on the
# HF config ``model_type`` so a ``base`` model's *actual* linear layers are
# offered rather than the generic Llama default. Config-only (no torch /
# no weights) — derived from each architecture's documented module names.
_ARCH_TARGET_MODULES: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    # Llama-family (gated MLP).
    "llama": _LLAMA_SHAPE,
    "mistral": _LLAMA_SHAPE,
    "mixtral": _LLAMA_SHAPE + ("w1", "w2", "w3"),
    "qwen2": _LLAMA_SHAPE,
    "qwen2_moe": _LLAMA_SHAPE,
    "qwen3": _LLAMA_SHAPE,
    "qwen3_moe": _LLAMA_SHAPE,
    "qwen4_exp_text": _QWEN4_EXP_TEXT_SHAPE,
    "gemma": _LLAMA_SHAPE,
    "gemma2": _LLAMA_SHAPE,
    "gemma3": _LLAMA_SHAPE,
    "gemma3_text": _LLAMA_SHAPE,
    "granite": _LLAMA_SHAPE,
    "granitemoe": _LLAMA_SHAPE + ("w1", "w2", "w3"),
    "cohere": _LLAMA_SHAPE,
    "deepseek_v3": _LLAMA_SHAPE,
    "stablelm": _LLAMA_SHAPE,
    "starcoder2": ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"),
    # Phi-family.
    "phi": ("q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"),
    "phi3": ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
    # GPT-2 / Conv1D style.
    "gpt2": ("c_attn", "c_proj", "c_fc"),
    "gptj": ("q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"),
    "gpt_neox": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    "falcon": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    "bloom": ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    "mpt": ("Wqkv", "out_proj", "up_proj", "down_proj"),
})


def _introspect_target_modules(base: str) -> Optional[Tuple[str, ...]]:
    """Return per-base target modules from a *cached* HF config, or ``None``.

    Loads ``AutoConfig`` with ``local_files_only=True`` (config-only — no
    torch, no weights, never a network download from a shell completer) and
    maps ``config.model_type`` onto :data:`_ARCH_TARGET_MODULES`. Returns
    ``None`` on any failure (transformers absent, model not cached, unknown
    arch) so the caller degrades to the canonical default. Never raises.
    """
    try:
        from transformers import AutoConfig  # lazy — keep CLI startup fast
    except ImportError:
        return None
    try:
        cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    except Exception:  # noqa: BLE001 — completer must never raise / hang
        return None
    text_config = getattr(cfg, "text_config", None)
    for candidate in (text_config, cfg):
        model_type = getattr(candidate, "model_type", None)
        if not isinstance(model_type, str) or not model_type:
            continue
        targets = _ARCH_TARGET_MODULES.get(model_type.lower())
        if targets is not None:
            return targets
    return None


def validate_shell(value: object) -> str:
    """Normalise + validate a shell name against ``SUPPORTED_SHELLS``."""
    if isinstance(value, bool):
        raise TypeError("shell must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"shell must be str, got {type(value).__name__}")
    if not value:
        raise ValueError("shell must be non-empty")
    if "\x00" in value:
        raise ValueError("shell must not contain null bytes")
    if len(value) > _MAX_SHELL_LEN:
        raise ValueError(f"shell name too long (> {_MAX_SHELL_LEN} chars)")
    normalised = value.lower().strip()
    if normalised not in SUPPORTED_SHELLS:
        allowed = ", ".join(sorted(SUPPORTED_SHELLS))
        raise ValueError(f"unknown shell {value!r}; known: {allowed}")
    return normalised


def _render_typer_script(shell: str) -> str:
    """Use the installed Typer's matching instruction and response formats."""
    from typer.completion import get_completion_script

    return get_completion_script(
        prog_name="soup", complete_var="_SOUP_COMPLETE", shell=shell
    ) + "\n"


def render_bash_script() -> str:
    """Render a bash completion script for the live Typer command tree."""
    return _render_typer_script("bash")


def render_zsh_script() -> str:
    """Render a zsh completion script for the live Typer command tree."""
    return _render_typer_script("zsh")


def render_fish_script() -> str:
    """Render a fish completion script for the live Typer command tree."""
    return _render_typer_script("fish")


def render_completion_script(shell: object) -> str:
    """Dispatch on shell name. Validates + renders one of the three scripts."""
    normalised = validate_shell(shell)
    if normalised == "bash":
        return render_bash_script()
    if normalised == "zsh":
        return render_zsh_script()
    if normalised == "fish":
        return render_fish_script()
    # Unreachable thanks to ``validate_shell``; defensive default.
    raise ValueError(f"unhandled shell {normalised!r}")


def complete_recipe_name(prefix: object) -> List[str]:
    """Suggest recipe names matching ``prefix`` (case-insensitive).

    Backed by ``soup_cli.recipes.catalog.list_recipes`` (lazy import).
    """
    if isinstance(prefix, bool):
        raise TypeError("prefix must be str, not bool")
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be str, got {type(prefix).__name__}")
    if "\x00" in prefix:
        # Defensive: shell completers should never raise.
        return []
    try:
        from soup_cli.recipes.catalog import RECIPES
    except ImportError:  # pragma: no cover
        return []
    p = prefix.lower()
    return [name for name in RECIPES if name.lower().startswith(p)]


def complete_target_modules(
    prefix: object,
    *,
    base: Optional[str] = None,
) -> List[str]:
    """Suggest ``target_modules`` values for the chosen ``base`` model.

    When ``base`` names a model whose HF config is in the local cache,
    v0.71.1 #210 introspects ``AutoConfig`` (config-only) and offers that
    architecture's *actual* linear-layer names (e.g. ``c_attn`` for GPT-2,
    ``query_key_value`` for Falcon). Falls back to the canonical Llama-shape
    defaults when ``base`` is omitted, transformers is unavailable, the model
    is not cached, or the architecture is unknown. The completer never raises
    and never hits the network.
    """
    if isinstance(prefix, bool):
        raise TypeError("prefix must be str, not bool")
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be str, got {type(prefix).__name__}")
    if base is not None and not isinstance(base, str):
        raise TypeError(f"base must be str | None, got {type(base).__name__}")
    if "\x00" in prefix:
        return []
    modules: Tuple[str, ...] = _DEFAULT_TARGET_MODULES
    if base:
        introspected = _introspect_target_modules(base)
        if introspected:
            modules = introspected
    return [m for m in modules if m.startswith(prefix)]


__all__ = [
    "SUPPORTED_SHELLS",
    "complete_recipe_name",
    "complete_target_modules",
    "render_bash_script",
    "render_completion_script",
    "render_fish_script",
    "render_zsh_script",
    "validate_shell",
]
