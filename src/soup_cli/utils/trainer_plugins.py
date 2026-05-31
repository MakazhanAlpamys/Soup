"""v0.45.0 Part D — Advanced trainer-plugin allowlist (schema-only).

Closed allowlist of optional trainer plugins so a future
``training.trainer_plugins: [grokfast, spectrum, ...]`` Pydantic field
can validate against a stable surface. Live wiring (callbacks, kernel
swaps, LLMCompressor passes) is deferred to v0.45.1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

_LOG = logging.getLogger("soup_cli.utils.trainer_plugins")

_PLUGIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
_MAX_DESCRIPTION = 256
_MAX_PLUGINS_PER_RUN = 8


@dataclass(frozen=True)
class TrainerPluginSpec:
    """One advanced trainer plugin descriptor."""

    name: str
    description: str
    required_package: Optional[str]


def _make(
    name: str,
    description: str,
    required_package: Optional[str],
) -> TrainerPluginSpec:
    if not _PLUGIN_NAME_RE.match(name):
        raise ValueError(
            "trainer-plugin name must be snake_case ([a-z0-9][a-z0-9_]{0,31})"
        )
    if not isinstance(description, str) or "\x00" in description:
        raise ValueError("description must be a NUL-free string")
    if len(description) > _MAX_DESCRIPTION:
        raise ValueError(f"description exceeds {_MAX_DESCRIPTION} chars")
    if required_package is not None:
        if not isinstance(required_package, str) or not required_package:
            raise ValueError("required_package must be a non-empty string")
    return TrainerPluginSpec(
        name=name, description=description, required_package=required_package
    )


_BUILTIN: Mapping[str, TrainerPluginSpec] = MappingProxyType(
    {
        "cce_plugin": _make(
            "cce_plugin",
            "Cut Cross-Entropy plugin variant (axolotl binding)",
            "cut-cross-entropy",
        ),
        "grokfast": _make(
            "grokfast",
            "Gradient-grokking accelerator (axolotl)",
            "grokfast",
        ),
        "spectrum": _make(
            "spectrum",
            "Gradient-norm-based layer freezing (axolotl)",
            None,
        ),
        "llmcompressor": _make(
            "llmcompressor",
            "Post-training compression (axolotl)",
            "llmcompressor",
        ),
        "sonicmoe": _make(
            "sonicmoe",
            "Alternative MoE kernel (axolotl)",
            None,
        ),
        "math_verify": _make(
            "math_verify",
            "Standalone math reward verifier (promoted v0.25.0)",
            None,
        ),
    }
)


def list_trainer_plugins() -> Mapping[str, TrainerPluginSpec]:
    """Return an immutable view of the registry."""
    return _BUILTIN


def get_trainer_plugin(name: str) -> TrainerPluginSpec:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    canonical = name.strip().lower()
    if not canonical or "\x00" in canonical:
        raise ValueError("name must be a non-empty NUL-free string")
    if canonical not in _BUILTIN:
        raise KeyError(canonical)
    return _BUILTIN[canonical]


def validate_trainer_plugin_list(names: Sequence[str]) -> Tuple[str, ...]:
    """Validate a list of trainer-plugin names. Returns canonical names."""
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise TypeError("names must be a list or tuple")
    if len(names) > _MAX_PLUGINS_PER_RUN:
        raise ValueError(
            f"too many trainer plugins (max {_MAX_PLUGINS_PER_RUN})"
        )
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        if not isinstance(raw, str):
            raise TypeError("plugin name must be a string")
        canonical = raw.strip().lower()
        if not canonical or "\x00" in canonical:
            raise ValueError("plugin name must be non-empty NUL-free")
        if canonical not in _BUILTIN:
            raise ValueError(
                f"unknown trainer plugin: {canonical!r}. supported: "
                f"{sorted(_BUILTIN)}"
            )
        if canonical in seen:
            raise ValueError(f"duplicate trainer plugin: {canonical!r}")
        seen.add(canonical)
        out.append(canonical)
    return tuple(out)


def _instantiate_cce_plugin() -> Any:
    """``cce_plugin`` — Cut Cross-Entropy advisory callback.

    The real CCE patching happens through the v0.28.0
    ``use_cut_ce`` flag (which patches model forward at trainer setup).
    The plugin surface returns a thin advisory callback so trainers can
    log that CCE is intended.
    """

    class _CCEAdvisoryCallback:
        plugin_name = "cce_plugin"

        def on_train_begin(self, args=None, state=None, control=None, **kwargs):  # noqa: ARG002
            # Logging is best-effort; trainer wraps this in a try/except.
            return control

    return _CCEAdvisoryCallback()


def _instantiate_simple_plugin(
    name: str, pip_package: str, attr_candidates: Sequence[str]
) -> Any:
    """Lazy-import an upstream plugin module + instantiate the first attr."""
    import importlib

    try:
        module = importlib.import_module(pip_package.replace("-", "_"))
    except ImportError as exc:
        raise ImportError(
            f"trainer plugin {name!r} requires the {pip_package!r} package. "
            f"Run: pip install {pip_package}"
        ) from exc
    for attr in attr_candidates:
        if hasattr(module, attr):
            cls_or_fn = getattr(module, attr)
            try:
                return cls_or_fn() if callable(cls_or_fn) else cls_or_fn
            except TypeError:
                # v0.53.7 L-F: narrow catch — only TypeError indicates "the
                # constructor signature doesn't match (caller must wire it
                # later)". Other exceptions surface a real bug in the
                # upstream module and should NOT be masked.
                return cls_or_fn
    # v0.53.7 L-A + L-E: no known attribute found. Logging a warning
    # surfaces that the upstream module's API may have drifted; returning
    # ``None`` lets callers filter rather than silently treating the bare
    # module as a callback (which crashes at trainer wiring time with a
    # confusing AttributeError).
    _LOG.warning(
        "trainer plugin %r: none of %s found on module %r; skipping",
        name, list(attr_candidates), pip_package,
    )
    return None


def _instantiate_plugin(name: str) -> Any:
    """Dispatch to the per-plugin instantiation helper."""
    if name == "cce_plugin":
        return _instantiate_cce_plugin()
    if name == "grokfast":
        return _instantiate_simple_plugin(
            "grokfast", "grokfast", ("GrokFastCallback", "Gradfilter", "gradfilter_ma")
        )
    if name == "spectrum":
        return _instantiate_simple_plugin(
            "spectrum", "spectrum-pytorch", ("SpectrumCallback", "Spectrum")
        )
    if name == "llmcompressor":
        return _instantiate_simple_plugin(
            "llmcompressor", "llmcompressor",
            ("LLMCompressorCallback", "LLMCompressor"),
        )
    if name == "sonicmoe":
        return _instantiate_simple_plugin(
            "sonicmoe", "sonicmoe", ("SonicMoECallback", "SonicMoE")
        )
    if name == "math_verify":
        return _instantiate_simple_plugin(
            "math_verify", "math-verify", ("MathVerifyCallback", "verify")
        )
    raise ValueError(f"unknown trainer plugin: {name!r}")  # pragma: no cover


def instantiate_trainer_plugins(names: Sequence[str]) -> Tuple[Any, ...]:
    """v0.53.6 #105 / v0.53.7 — instantiate live upstream callbacks.

    Validates ``names`` against :func:`validate_trainer_plugin_list`, then
    lazy-imports each plugin's pip package and returns a tuple of
    instantiated callbacks (or modules for plugins whose constructor needs
    trainer-specific kwargs the caller will supply).

    Missing pip deps surface as :class:`ImportError` with a friendly
    ``pip install <pkg>`` hint (Rich-escape applied at the CLI boundary —
    this helper returns the raw exception so callers can decide).

    Raises:
        TypeError: per :func:`validate_trainer_plugin_list`.
        ValueError: per :func:`validate_trainer_plugin_list`.
        ImportError: when a plugin's required pip package is not installed.
    """
    canonical = validate_trainer_plugin_list(names)
    # v0.53.7 L-A: filter out ``None`` from helpers that could not find an
    # API surface in the upstream module (logged at WARNING in
    # ``_instantiate_simple_plugin``).
    instances = (_instantiate_plugin(n) for n in canonical)
    return tuple(plugin for plugin in instances if plugin is not None)


__all__ = [
    "TrainerPluginSpec",
    "list_trainer_plugins",
    "get_trainer_plugin",
    "validate_trainer_plugin_list",
    "instantiate_trainer_plugins",
]
