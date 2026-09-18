"""Config values Soup still loads, with a warning, for one release before refusing them.

A value that never took effect is a defect to remove, but refusing it at once
breaks the configs Soup itself wrote while that value was the default:
``soup autopilot`` and every run-config dump write the whole model, defaults
included. So such a value loads with a warning naming the release that will
refuse it, the same staging #627 used for unknown keys (#759, #795).

A schema validator calls :func:`warn_deprecated_value`; the loader turns each
warning into one printed line per load. Code that builds ``SoupConfig``
directly still sees an ordinary :class:`SoupConfigDeprecationWarning`.
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
#: ships in 0.75, the refusal in 0.76.
DEPRECATED_VALUE_REJECTION_VERSION = "0.76"


class SoupConfigDeprecationWarning(FutureWarning):
    """A config value that loads today and will be refused in a named release."""


def deadline_clause() -> str:
    """The sentence every deprecation message ends with."""
    return f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."


def warn_deprecated_value(message: str) -> None:
    """Warn that ``message`` describes a value that will be refused."""
    warnings.warn(
        f"{message} {deadline_clause()}", SoupConfigDeprecationWarning, stacklevel=3
    )
