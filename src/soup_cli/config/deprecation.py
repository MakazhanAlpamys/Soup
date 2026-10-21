"""Config values Soup still loads, with a warning, for one release before refusing them.

A value that never took effect is a defect to remove, but refusing it at once
breaks the configs Soup itself wrote while that value was the default:
``soup autopilot`` and every run-config dump write the whole model, defaults
included. So such a value loads with a warning naming the release that will
refuse it, the same staging #627 used for unknown keys (#759, #795).

A schema validator calls :func:`warn_deprecated_value`; the loader turns each
warning into one printed line per load. Code that builds ``SoupConfig``
directly still sees an ordinary :class:`SoupConfigDeprecationWarning`. Once
:data:`DEPRECATED_VALUE_SEVERITY` is ``"error"`` the same call raises
``ValueError`` instead, so the value is refused wherever it is validated.
"""

from __future__ import annotations

import warnings

#: The release that stops warning about these values and refuses them.
#:
#: Written out once, here, and read by every warning message, because a
#: duplicated version is how a warning keeps promising a refusal after the
#: refusal shipped. The deadline test compares it against
#: ``soup_cli.__version__``, so the release that reaches it turns a test red
#: instead of turning the warning into a lie. One minor of notice: the warning
#: ships in 0.76, the refusal in 0.77.
DEPRECATED_VALUE_REJECTION_VERSION = "0.77"

#: What to do about a deprecated config value (#759, #795, #1418).
#:
#: ``"warn"``  -- load it, ignore it, and print one line naming the release
#:               that will refuse it.
#: ``"error"`` -- refuse it: the config load raises, and ``soup ab --beta``
#:               exits with a usage error.
#:
#: Mirrors ``STAGED_FIELD_SEVERITY`` in ``config/loader.py``. Ships ``"warn"``;
#: the release commit that bumps ``__version__`` to
#: :data:`DEPRECATED_VALUE_REJECTION_VERSION` flips this to ``"error"``, together
#: with ``STAGED_FIELD_SEVERITY``. Read it through
#: :func:`refuses_deprecated_values`, at call time, never copied at import.
DEPRECATED_VALUE_SEVERITY = "warn"


class SoupConfigDeprecationWarning(FutureWarning):
    """A config value that loads today and will be refused in a named release."""


def refuses_deprecated_values() -> bool:
    """Whether a deprecated value is refused rather than warned about."""
    return DEPRECATED_VALUE_SEVERITY == "error"


def deadline_clause() -> str:
    """The sentence every deprecation message ends with, in either severity."""
    if refuses_deprecated_values():
        return f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
    return f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."


def warn_deprecated_value(message: str, *, refusal: "str | None" = None) -> None:
    """Warn that ``message`` describes a value that will be refused, or refuse it.

    Under ``"warn"`` this warns with ``message``. Under ``"error"`` it raises
    ``ValueError`` with ``refusal``, or with ``message`` when no refusal text is
    given: pass one whenever ``message`` says the value "is ignored", which is
    no longer true once it is refused. Either text should end with what the
    user does instead; the deadline clause is appended to both.
    """
    if refuses_deprecated_values():
        text = message if refusal is None else refusal
        raise ValueError(f"{text} {deadline_clause()}")
    warnings.warn(
        f"{message} {deadline_clause()}", SoupConfigDeprecationWarning, stacklevel=3
    )
