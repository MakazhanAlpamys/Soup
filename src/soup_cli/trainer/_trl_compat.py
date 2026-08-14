"""Version-tolerant access to the ``trl`` APIs the preference trainers use.

``trl`` has broken Soup twice in the same place, and both times the diagnosis
was wrong because it was made by *reading source*. The rule this earned: a
version bound derived by reading source is a hypothesis; the experiment that
settles it is CONSTRUCTING THE OBJECT.

Two independent things move, and they move on different schedules:

**1. ``max_prompt_length`` was removed from the preference configs, in stages.**
Measured by constructing each config with the exact keyword the wrappers pass:

    trl              dpo   kto   orpo   cpo   bco
    0.14.0 - 0.26.2  yes   yes   yes    yes   yes
    0.27.0 - 0.27.2  yes   NO    yes    yes   yes
    0.28.0           yes   NO    NO     NO    NO
    0.29.0 - 1.9.2   NO    NO    NO     NO    NO

There is no successor field. ``max_length`` survives on every version and
prompt truncation is folded into it, so the migration is simply to stop
passing the keyword to a config that no longer has it — NOT to rename it.

**2. Three configs left the public ``trl`` namespace at 0.29.0.**
``ORPOConfig`` / ``CPOConfig`` / ``BCOConfig`` and their trainers were not
deleted; they live on under ``trl.experimental.<algo>``. That break is harder
than a rejected keyword — it is an ``ImportError`` at ``setup()``, so the task
dies before it can report anything useful.

Both helpers below are deliberately *capability* probes rather than version
comparisons. A version check re-encodes the very table that was wrong twice;
asking the installed object what it accepts cannot go stale.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def _installed_trl_version() -> str:
    """Best-effort version string, for error messages only."""
    try:
        import trl

        return str(getattr(trl, "__version__", "unknown"))
    except Exception:  # pragma: no cover - trl ships in the [train] extra
        return "unknown"


def resolve_trl_symbol(name: str, experimental_module: str | None = None) -> Any:
    """Return ``trl.<name>``, falling back to its ``trl.experimental`` home.

    The public namespace is tried FIRST, so any ``trl`` that still exports the
    symbol keeps the supported, non-experimental path and no warning is
    emitted. The fallback only engages on a ``trl`` that has already moved it.

    ``getattr`` rather than ``hasattr``: ``trl`` exposes these through a lazy
    module, so a submodule that fails to import raises something other than
    ``AttributeError`` and ``hasattr`` would report a clean ``False`` without
    saying why. The original exception is chained onto the raised
    ``ImportError`` rather than swallowed.
    """
    import trl

    public_error: Exception | None = None
    try:
        return getattr(trl, name)
    except Exception as exc:  # noqa: BLE001 - chained below, never swallowed
        public_error = exc

    experimental_error: Exception | None = None
    if experimental_module:
        try:
            return getattr(importlib.import_module(experimental_module), name)
        except Exception as exc:  # noqa: BLE001 - chained below
            experimental_error = exc

    where = f" nor in {experimental_module}" if experimental_module else ""
    raise ImportError(
        f"installed trl {_installed_trl_version()} does not provide {name!r} "
        f"in the `trl` namespace{where}. The [train] extra's version bounds and "
        f"this trainer disagree — see pyproject.toml. "
        f"(trl.{name}: {type(public_error).__name__}: {public_error}"
        + (
            f"; {experimental_module}.{name}: "
            f"{type(experimental_error).__name__}: {experimental_error})"
            if experimental_module
            else ")"
        )
    ) from public_error


def config_accepts(config_cls: type, field: str) -> bool:
    """Does this trl config class actually take ``field`` as a keyword?

    Asked of the class the caller is about to construct, so it stays correct
    across a namespace move as well as a version bump. Configs are dataclasses,
    so the generated ``__init__`` signature is the authoritative list of
    accepted keywords — including the ones inherited from ``TrainingArguments``.
    """
    try:
        return field in inspect.signature(config_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level __init__
        return False


def prompt_length_kwargs(config_cls: type, max_prompt_length: int) -> dict[str, int]:
    """``{'max_prompt_length': N}`` iff this trl's config still accepts it.

    Returns an empty dict on a trl that removed the field, which is the whole
    migration: there is no replacement keyword to pass instead. Behaviour on
    every trl that still has the field is byte-identical to passing it
    directly, so nothing changes for the versions Soup already supported.
    """
    if config_accepts(config_cls, "max_prompt_length"):
        return {"max_prompt_length": max_prompt_length}
    return {}
