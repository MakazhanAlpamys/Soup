"""v0.53.6 #101 — Soup plugin TrainerCallback.

Bridges :mod:`soup_cli.plugins` registered hooks into the HF Trainer
callback surface. For every enabled plugin (via :func:`list_plugins` +
``spec.enabled``), discovers implemented hooks via :func:`discover_hooks`
and dispatches the matching trainer event.

Per-plugin hook exceptions are swallowed at WARNING level — one
misbehaving plugin must not crash a multi-hour training run. This
mirrors the v0.44.0 / v0.45.0 plugin loader policy.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _collect_active_hooks() -> list[tuple[str, dict[str, Any]]]:
    """Return ``[(plugin_name, hook_map), ...]`` for every enabled plugin.

    Lazy-imports :mod:`soup_cli.plugins` so the callback module stays
    cheap to import in CI / non-training contexts.
    """
    from soup_cli.plugins import discover_hooks, list_plugins

    out: list[tuple[str, dict[str, Any]]] = []
    for name, spec in list_plugins().items():
        if not spec.enabled:
            continue
        hooks = discover_hooks(spec.plugin)
        if hooks:
            out.append((name, hooks))
    return out


def _safe_invoke(
    plugin_name: str, hook_name: str, hook: Any, ctx: dict[str, Any]
) -> None:
    try:
        hook(ctx)
    except Exception:  # noqa: BLE001 — plugin failure must not crash training
        logger.warning(
            "Plugin %r hook %r raised; continuing",
            plugin_name,
            hook_name,
            exc_info=True,
        )


def _build_callback_class() -> type:
    """Construct the ``SoupPluginCallback`` class with transformers as parent.

    Lazy-imports :mod:`transformers` so the wiring helper can be imported
    in CI without the heavy dep installed.
    """
    from transformers import TrainerCallback

    class SoupPluginCallback(TrainerCallback):
        """Fans HF Trainer events out to every enabled Soup plugin."""

        def __init__(
            self, hooks: list[tuple[str, dict[str, Any]]] | None = None
        ) -> None:
            super().__init__()
            # Snapshot at construction time so a plugin registered MID-run
            # does not silently start receiving hooks halfway through. The
            # caller may pre-collect hooks (the ``build_plugin_callback``
            # path) to avoid a redundant registry scan + close a tiny
            # race-window between "is any plugin enabled?" and
            # "snapshot the registry".
            self._hooks = (
                list(hooks) if hooks is not None else _collect_active_hooks()
            )

        def _dispatch(self, hook_name: str, context: dict[str, Any]) -> None:
            for plugin_name, hooks in self._hooks:
                hook = hooks.get(hook_name)
                if hook is None:
                    continue
                _safe_invoke(plugin_name, hook_name, hook, context)

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
            self._dispatch(
                "pre_train", {"args": args, "state": state, "control": control}
            )

        def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
            self._dispatch(
                "post_train", {"args": args, "state": state, "control": control}
            )

        def on_step_begin(self, args, state, control, **kwargs):  # noqa: D401
            self._dispatch(
                "pre_step", {"args": args, "state": state, "control": control}
            )

        def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
            self._dispatch(
                "post_step", {"args": args, "state": state, "control": control}
            )

    return SoupPluginCallback


def build_plugin_callback() -> Any:
    """Return a new ``SoupPluginCallback`` instance, or ``None`` if no
    enabled plugins implement any hook (no-op short-circuit).

    Collects hooks ONCE and passes the snapshot to the callback so a
    plugin registered between the "is empty?" check and the constructor
    cannot silently slip into the active hook list — review fix.
    """
    hooks = _collect_active_hooks()
    if not hooks:
        return None
    callback_cls = _build_callback_class()
    return callback_cls(hooks)


__all__ = ["build_plugin_callback"]
