"""Chat-template registry + override (v0.36.0 Part C).

Replaces the silent ``f"{role}: {content}"`` fallback in
``trainer/sft.py`` with an explicit registry of named chat templates plus a
``DataConfig.chat_template`` override field that accepts either a registered
name or a raw Jinja string.

Mirrors LlamaFactory and Axolotl behaviour: tokenizer without a chat template
+ no override = hard error. Silent garbage labels are no longer possible.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional

# Jinja templates for popular chat formats. Kept minimal — full upstream
# templates ship with the model tokenizer; these are conservative fallbacks
# for users explicitly opting in via DataConfig.chat_template = "<name>".
#
# All templates assume ``messages`` is a list of ``{"role", "content"}``
# dicts and tolerate an optional leading ``system`` turn.

_CHATML = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_LLAMA3 = (
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

# Mistral's official template injects the system prompt INSIDE the first
# [INST] block, not as a freestanding turn. We track whether we've emitted
# the leading [INST] yet and prepend the system content to the next user
# turn's content.
_MISTRAL = (
    "{% set system = namespace(content='') %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% set system.content = message['content'] %}"
    "{% elif message['role'] == 'user' %}"
    "{% if system.content %}"
    "[INST] {{ system.content }}\n\n{{ message['content'] }} [/INST]"
    "{% set system.content = '' %}"
    "{% else %}"
    "[INST] {{ message['content'] }} [/INST]"
    "{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}</s>"
    "{% endif %}"
    "{% endfor %}"
)

_GEMMA3 = (
    "{% for message in messages %}"
    "<start_of_turn>{{ 'user' if message['role'] == 'user' else 'model' }}\n"
    "{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

_DEEPSEEK_R1 = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<｜User｜>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜>{% endif %}"
)

# Phi-4 and Qwen2.5 both use a ChatML variant — re-use the ChatML template.
# Wrap in MappingProxyType so callers cannot mutate the registry at runtime.
_REGISTRY: "MappingProxyType[str, str]" = MappingProxyType({
    "chatml": _CHATML,
    "qwen2.5": _CHATML,
    "qwen": _CHATML,
    "phi4": _CHATML,
    "phi-4": _CHATML,
    "llama3": _LLAMA3,
    "llama-3": _LLAMA3,
    "mistral": _MISTRAL,
    "gemma3": _GEMMA3,
    "gemma-3": _GEMMA3,
    "deepseek-r1": _DEEPSEEK_R1,
})

# Treat anything containing Jinja control tokens (`{%` / `{{`) as a raw
# Jinja string instead of a registry key.
_JINJA_MARKERS = ("{%", "{{")


def list_template_names() -> list[str]:
    """Return the canonical (sorted) list of registered template names."""
    return sorted(_REGISTRY.keys())


def get_template(name: str) -> str:
    """Look up a registered template by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"chat_template '{name}' is not registered. "
            f"Known: {', '.join(list_template_names())}"
        )
    return _REGISTRY[name]


def _looks_like_jinja(value: str) -> bool:
    return any(marker in value for marker in _JINJA_MARKERS)


def resolve_chat_template(value: Optional[str]) -> Optional[str]:
    """Resolve a ``DataConfig.chat_template`` value to a Jinja string.

    - ``None`` / empty → ``None``
    - Looks-like-Jinja → returned unchanged
    - Registered name → registry lookup
    - Otherwise → ``KeyError`` (typo in the name)
    """
    if not value:
        return None
    if _looks_like_jinja(value):
        return value
    return get_template(value)


def apply_chat_template_override(
    tokenizer: Any, value: Optional[str], console: Any | None = None
) -> bool:
    """Set ``tokenizer.chat_template`` from a name or Jinja string.

    No-op when ``value`` is ``None`` / empty. Mutates the tokenizer in-place
    so downstream calls (HF ``apply_chat_template`` and the tokenizer's
    ``.save_pretrained``) pick up the override.

    Returns ``True`` when an override was applied. When ``console`` is
    supplied and an override fires, prints a yellow advisory so the user
    knows that ``soup push`` will persist the override into
    ``tokenizer_config.json``.
    """
    resolved = resolve_chat_template(value)
    if resolved is None:
        return False
    tokenizer.chat_template = resolved
    if console is not None:
        console.print(
            "[yellow]chat_template override applied.[/] Subsequent "
            "tokenizer.save_pretrained() / soup push will persist this "
            "Jinja string into tokenizer_config.json — replacing whatever "
            "the model originally shipped."
        )
    return True
