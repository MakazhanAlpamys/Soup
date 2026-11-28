"""``--trust-remote-code`` opt-in (v0.36.0 Part B).

Replaces the previous unconditional ``trust_remote_code=True`` in
sft.py / chat.py / serve.py / etc. with an explicit, auditable flag plus a
trusted-org allowlist that suppresses warning noise on first-party models.

Design:

- ``is_known_safe(model_name)``: true when the repo id starts with a known
  org prefix that does NOT ship custom modeling code (Meta, Mistral, Qwen,
  Google, etc.). Local paths and unknown orgs return False.
- ``model_requires_trust_remote_code(model_or_path)``: best-effort probe of
  ``config.json`` for ``auto_map``. Export can allow built-in config classes.
- ``tokenizer_requires_trust_remote_code(model_or_path)``: independent local
  probe for a custom tokenizer mapping without an installed fallback.
- ``resolve_trust_remote_code(model_name, requested, console, requires_remote_code)``:
  the gate. Returns ``True`` / ``False`` for the kwarg, or raises ``ValueError``
  with an actionable message when the model needs custom code but the user
  did not opt in.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from rich.console import Console

# Known organisations that do NOT ship custom modeling code with their HF
# checkpoints. Adding a prefix here suppresses the warning panel when the
# user passes ``--trust-remote-code`` against a safe org.
KNOWN_SAFE_PREFIXES: tuple[str, ...] = (
    "meta-llama/",
    "mistralai/",
    "Qwen/",
    "google/",
    "microsoft/",
    "deepseek-ai/",
    "01-ai/",
    "tiiuae/",
    "HuggingFaceH4/",
    "openai-community/",
    "facebook/",
    "EleutherAI/",
    "CohereForAI/",
    "stabilityai/",
    "ibm-granite/",
)


def is_known_safe(model_name: Any) -> bool:
    """Return True if ``model_name`` is a Hub repo id under a known-safe org prefix.

    The allowlist answers for Hub repo ids only. A name that exists on disk,
    relative to the working directory, is a local path: that is how
    ``from_pretrained`` reads it, so an existing path wins over a Hub id of the
    same spelling. A local directory such as ``Qwen/my-merge`` is therefore
    never known-safe, and ``--trust-remote-code`` keeps its warning panel.
    """
    if not isinstance(model_name, str) or not model_name:
        return False
    if os.path.exists(model_name):
        return False
    return any(model_name.startswith(prefix) for prefix in KNOWN_SAFE_PREFIXES)


def _read_local_metadata(model_or_path: str, filename: str) -> Optional[dict[str, Any]]:
    """Read a JSON object from a local model directory, if available."""
    if not isinstance(model_or_path, str) or not os.path.isdir(model_or_path):
        return None
    try:
        config_path = os.path.join(model_or_path, filename)
        if not os.path.isfile(config_path):
            return None
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_builtin_model_type(model_type: Any) -> bool:
    """Check the installed Transformers registry without loading a model class."""
    if not isinstance(model_type, str) or not model_type:
        return False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:
        # An unavailable registry leaves the custom-code requirement in place.
        return False
    return model_type in CONFIG_MAPPING_NAMES


def model_requires_trust_remote_code(
    model_or_path: str, *, allow_builtin_config: bool = False,
) -> Optional[bool]:
    """Return whether local config metadata signals that custom code is required.

    By default any nonempty ``auto_map`` signals custom code. Export opts into
    the built-in config exception and passes the requested trust flag to its
    Transformers loaders. Missing or unreadable metadata returns ``None``.
    """
    data = _read_local_metadata(model_or_path, "config.json")
    if data is None:
        return None
    return bool(data.get("auto_map")) and not (
        allow_builtin_config and _is_builtin_model_type(data.get("model_type"))
    )


def _has_native_tokenizer(model_or_path: str, tokenizer_class: Any) -> bool:
    """Check the installed AutoTokenizer registry without loading a checkpoint."""
    config = _read_local_metadata(model_or_path, "config.json") or {}
    model_type = config.get("model_type")
    try:
        from transformers.models.auto.tokenization_auto import (
            MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
            TOKENIZER_MAPPING_NAMES,
            tokenizer_class_from_name,
        )
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if isinstance(model_type, str):
            # Transformers ignores remote tokenizer mappings for these types with trust off.
            if model_type in MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS:
                return True
            if TOKENIZER_MAPPING_NAMES.get(model_type) and _is_builtin_model_type(model_type):
                return True
        if isinstance(tokenizer_class, str) and tokenizer_class:
            name = tokenizer_class.removesuffix("Fast")
            cls = tokenizer_class_from_name(name) or tokenizer_class_from_name(name + "Fast")
            return isinstance(cls, type) and issubclass(cls, PreTrainedTokenizerBase)
    except Exception:
        # If the installed registry cannot resolve a fallback, retain the opt-in requirement.
        return False
    return False


def tokenizer_requires_trust_remote_code(model_or_path: str) -> bool:
    """Return whether local tokenizer metadata declares only a custom tokenizer."""
    data = _read_local_metadata(model_or_path, "tokenizer_config.json")
    if data is None:
        return False
    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict):
        mapping = auto_map.get("AutoTokenizer")
    else:
        mapping = auto_map if isinstance(auto_map, list) else None
    return bool(mapping) and not _has_native_tokenizer(model_or_path, data.get("tokenizer_class"))


def resolve_trust_remote_code(
    model_name: str,
    requested: bool,
    console: "Console | None" = None,
    requires_remote_code: bool = False,
    *,
    component: Literal["model", "tokenizer"] = "model",
) -> bool:
    """Decide whether to pass ``trust_remote_code=True`` to HF loaders.

    Args:
        model_name: HF repo id or local path.
        requested: ``True`` when the user passed ``--trust-remote-code``.
        console: Rich Console for the warning panel (optional).
        requires_remote_code: ``True`` when the probe indicates custom code.
            Note: ``model_requires_trust_remote_code`` only probes LOCAL
            paths; HF Hub repo IDs return ``None`` (unknown) which the
            caller normally coerces to ``False``. In the unknown-Hub-path
            case HF's own ``from_pretrained`` will still raise loudly when
            it actually needs custom code, so the gate is defence-in-depth
            rather than the only line of defence.
        component: Which loader the error and warning refer to.

    Returns:
        ``bool`` to pass directly as ``trust_remote_code=...``.

    Raises:
        ValueError: model needs custom code but user did not opt in, or
            ``model_name`` is empty.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")

    if not requested and requires_remote_code:
        if component == "tokenizer":
            raise ValueError(
                f"Tokenizer {model_name} requires trust_remote_code=True (custom tokenizer "
                f"code via auto_map). Re-run with --trust-remote-code if you trust the source."
            )
        raise ValueError(
            f"Model {model_name} requires trust_remote_code=True (custom "
            f"modeling code via auto_map). Re-run with --trust-remote-code "
            f"if you trust the source. This change in v0.36.0 makes the "
            f"opt-in explicit; it was previously enabled by default."
        )

    if requested and not is_known_safe(model_name) and console is not None:
        # Lazy import — keeps `import soup_cli.utils.trust_remote` cheap
        # in environments that don't otherwise pull in rich.
        from rich.panel import Panel
        from rich.text import Text

        from soup_cli.utils.terminal import strip_control

        source = "the tokenizer files" if component == "tokenizer" else "the model repo"
        console.print(
            Panel.fit(
                Text.assemble(
                    ("--trust-remote-code is enabled for", "yellow"),
                    " ",
                    (strip_control(model_name), "bold"),
                    f"\nThis will execute Python code shipped in {source}. "
                    "Only proceed if you trust the source.",
                ),
                title="[red]REMOTE CODE WARNING[/]",
                border_style="red",
            )
        )

    return bool(requested)
